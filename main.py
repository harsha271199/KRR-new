"""CLI entry point for the KRR Fact Verification System.

Usage examples:
    python main.py --pipeline krr
    python main.py --pipeline baseline --env-file .env.prod
    python main.py --pipeline both --output-json results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with attributes: pipeline, env_file, output_json.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="KRR Fact Verification System — run KRR and/or Baseline RAG pipelines.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["krr", "baseline", "both"],
        default="both",
        help="Which pipeline(s) to run (default: both).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        metavar="PATH",
        help="Path to the .env configuration file (default: .env).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        help=(
            "Override JSON_OUTPUT_PATH from config: write per-claim KRR results "
            "to this file (newline-delimited JSON)."
        ),
    )
    return parser.parse_args(argv)


def _build_llm_backend(config):
    """Instantiate the configured LLM backend.

    Args:
        config: A :class:`~config.Config` instance.

    Returns:
        An :class:`~baseline.llm.LLMBackend` implementation.
    """
    from baseline.llm import HuggingFaceLLM, OpenAILLM

    if config.llm_backend == "openai":
        logger.info("Initializing OpenAI LLM backend (model=%s).", config.openai_model)
        return OpenAILLM(config)
    else:
        logger.info(
            "Initializing HuggingFace LLM backend (model=%s).", config.hf_model_name
        )
        return HuggingFaceLLM(config)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the KRR Fact Verification CLI.

    Args:
        argv: Optional argument list for testing; defaults to sys.argv[1:].

    Returns:
        Exit code (0 on success, 1 on fatal error).
    """
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    try:
        from config import Config, ConfigError

        config = Config.from_env(env_file=args.env_file)
        logger.info("Configuration loaded from %r.", args.env_file)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error loading configuration: %s", exc)
        return 1

    # Override JSON output path if supplied on the command line.
    if args.output_json is not None:
        # Config is a frozen-ish dataclass but json_output_path is mutable.
        object.__setattr__(config, "json_output_path", args.output_json)

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    try:
        from data.loader import DataLoader

        logger.info("Loading FEVER dataset from %r.", config.fever_dataset_path)
        loader = DataLoader(config)
        records = loader.load()
        logger.info("Loaded %d records.", len(records))
    except FileNotFoundError as exc:
        logger.error("Dataset file not found: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load dataset: %s", exc)
        return 1

    if not records:
        logger.warning("No records loaded — nothing to evaluate.")
        return 0

    ground_truth = [record.label for record in records]
    claims = [record.claim for record in records]

    # ------------------------------------------------------------------
    # 3. Build / load retrieval index
    # ------------------------------------------------------------------
    try:
        from retrieval.retriever import Retriever

        retriever = Retriever(config)

        if config.index_path and Path(config.index_path).exists():
            logger.info("Loading pre-built index from %r.", config.index_path)
            retriever.load_index()
        else:
            logger.info("Building retrieval index (method=%s).", config.retrieval_method)
            corpus = loader.load_corpus()
            retriever.build_index(corpus)
            logger.info("Index built over %d sentences.", len(corpus))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize retriever: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # 4. Initialize KRR-specific components (if needed)
    # ------------------------------------------------------------------
    krr_pipeline = None
    if args.pipeline in ("krr", "both"):
        try:
            from knowledge.extractor import TripleExtractor
            from reasoning.reasoner import ReasoningModule
            from pipeline.krr import KRRPipeline

            logger.info(
                "Initializing TripleExtractor (spaCy model=%s).", config.spacy_model
            )
            extractor = TripleExtractor(config)
            reasoner = ReasoningModule()
            krr_pipeline = KRRPipeline(
                retriever=retriever,
                extractor=extractor,
                reasoner=reasoner,
                config=config,
            )
            logger.info("KRR pipeline ready.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize KRR pipeline: %s", exc)
            return 1

    # ------------------------------------------------------------------
    # 5. Initialize Baseline pipeline (if needed)
    # ------------------------------------------------------------------
    baseline_pipeline = None
    if args.pipeline in ("baseline", "both"):
        try:
            from baseline.pipeline import BaselinePipeline

            llm_backend = _build_llm_backend(config)
            baseline_pipeline = BaselinePipeline(
                retriever=retriever,
                llm_backend=llm_backend,
                config=config,
            )
            logger.info("Baseline RAG pipeline ready.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize Baseline pipeline: %s", exc)
            return 1

    # ------------------------------------------------------------------
    # 6. Run pipeline(s) over all records
    # ------------------------------------------------------------------
    from models import BaselineResult, KRRResult

    krr_results: list[KRRResult] = []
    baseline_results: list[BaselineResult] = []

    total = len(claims)

    for i, (claim, record) in enumerate(zip(claims, records), 1):
        if i % max(1, total // 10) == 0 or i == total:
            logger.info("Processing claim %d / %d …", i, total)

        if krr_pipeline is not None:
            krr_result = krr_pipeline.run(claim)
            krr_results.append(krr_result)

        if baseline_pipeline is not None:
            baseline_result = baseline_pipeline.run(claim)
            baseline_results.append(baseline_result)

    # ------------------------------------------------------------------
    # 7. Pad missing results so Evaluator lengths align
    # ------------------------------------------------------------------
    # If only one pipeline was run, create placeholder results for the other
    # so the Evaluator receives equal-length lists.
    if krr_pipeline is None:
        # Baseline-only run: create dummy KRR results
        krr_results = [
            KRRResult(
                claim=r.claim,
                retrieved_evidence=[],
                claim_triples=[],
                evidence_triples=[],
                support_count=0,
                refute_count=0,
                irrelevant_count=0,
                verdict="NOT ENOUGH INFO",
                attribution=[],
                error="KRR pipeline not run",
            )
            for r in baseline_results
        ]

    if baseline_pipeline is None:
        # KRR-only run: create dummy Baseline results
        from models import BaselineResult

        baseline_results = [
            BaselineResult(
                claim=r.claim,
                retrieved_evidence=[],
                prompt="",
                verdict="NOT ENOUGH INFO",
                raw_llm_response="",
                error="Baseline pipeline not run",
            )
            for r in krr_results
        ]

    # ------------------------------------------------------------------
    # 8. Evaluate and write report
    # ------------------------------------------------------------------
    try:
        from evaluation.evaluator import Evaluator

        evaluator = Evaluator()
        report = evaluator.evaluate(
            krr_results=krr_results,
            baseline_results=baseline_results,
            ground_truth=ground_truth,
        )
        evaluator.save_report(report, path=config.eval_output_path)
        logger.info("Evaluation report written to %r.", config.eval_output_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
