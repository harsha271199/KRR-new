"""Reasoning module: compares claim triples against evidence triples and assigns a verdict."""

from __future__ import annotations

from models import ReasoningResult, Triple


class ReasoningModule:
    """Explicitly compares claim triples against evidence triples.

    For each evidence triple, the module checks whether it supports, refutes,
    or is irrelevant to the claim triples, then assigns a verdict with full
    attribution to the triples that drove the decision.
    """

    def reason(
        self,
        claim_triples: list[Triple],
        evidence_triples: list[Triple],
    ) -> ReasoningResult:
        """Compute support/refute/irrelevant counts and assign verdict.

        Args:
            claim_triples: Structured triples extracted from the claim sentence.
            evidence_triples: Structured triples extracted from evidence sentences.

        Returns:
            A ReasoningResult with counts, verdict, and attribution triples.
        """
        if not isinstance(claim_triples, list):
            raise TypeError(
                f"claim_triples: expected list[Triple], got {type(claim_triples).__name__}"
            )
        if not isinstance(evidence_triples, list):
            raise TypeError(
                f"evidence_triples: expected list[Triple], got {type(evidence_triples).__name__}"
            )

        # Requirements 5.8 and 5.9: empty inputs → NOT ENOUGH INFO
        if not claim_triples or not evidence_triples:
            return ReasoningResult(
                support_count=0,
                refute_count=0,
                irrelevant_count=0,
                verdict="NOT ENOUGH INFO",
                attribution=[],
            )

        support_count = 0
        refute_count = 0
        irrelevant_count = 0
        supporting_triples: list[Triple] = []
        refuting_triples: list[Triple] = []

        for evidence_triple in evidence_triples:
            # Find claim triples that share subject and relation with this evidence triple
            matching_claims = [
                c
                for c in claim_triples
                if c.subject == evidence_triple.subject
                and c.relation == evidence_triple.relation
            ]

            if not matching_claims:
                # Requirement 5.3: no subject+relation overlap → irrelevant
                irrelevant_count += 1
            else:
                # Check object match against each matching claim triple
                for claim_triple in matching_claims:
                    if evidence_triple.object == claim_triple.object:
                        # Requirement 5.1: full match → support
                        support_count += 1
                        supporting_triples.append(evidence_triple)
                    else:
                        # Requirement 5.2: subject+relation match but object differs → refute
                        refute_count += 1
                        refuting_triples.append(evidence_triple)

        # Verdict assignment (Requirements 5.4, 5.5, 5.6)
        # Tie-breaking: support_count == refute_count > 0 → NOT ENOUGH INFO (Req 5.6 / design 4.5)
        if support_count > refute_count and support_count > 0:
            verdict = "SUPPORTS"
            attribution = supporting_triples
        elif refute_count > support_count and refute_count > 0:
            verdict = "REFUTES"
            attribution = refuting_triples
        else:
            verdict = "NOT ENOUGH INFO"
            attribution = []

        return ReasoningResult(
            support_count=support_count,
            refute_count=refute_count,
            irrelevant_count=irrelevant_count,
            verdict=verdict,
            attribution=attribution,
        )
