"""In-memory knowledge graph with namespace support.

Stores KnowledgeRecord objects keyed by namespace, with deduplication
and flexible retrieval by namespace, subject, or relation.
"""

from __future__ import annotations

from models import KnowledgeRecord, Triple


class KnowledgeGraph:
    """Namespaced in-memory store for knowledge triples.

    Triples are stored as ``KnowledgeRecord`` objects grouped by namespace
    (e.g. ``"claim"`` or ``"evidence"``).  Exact-match duplicates within the
    same namespace are silently ignored.
    """

    def __init__(self) -> None:
        # namespace -> list of records
        self._records: dict[str, list[KnowledgeRecord]] = {}
        # namespace -> set of (subject, relation, object) tuples for dedup
        self._seen: dict[str, set[tuple[str, str, str]]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_triple(
        self,
        triple: Triple,
        namespace: str,
        source_sentence: str = "",
    ) -> None:
        """Add a triple to the graph under *namespace*.

        If an identical ``(subject, relation, object)`` triple already exists
        in *namespace*, this call is a no-op.

        Args:
            triple: The ``Triple`` to store.
            namespace: Logical partition (e.g. ``"claim"`` or ``"evidence"``).
            source_sentence: The sentence the triple was extracted from.
        """
        key = (triple.subject, triple.relation, triple.object)

        if namespace not in self._seen:
            self._seen[namespace] = set()
            self._records[namespace] = []

        if key in self._seen[namespace]:
            return

        self._seen[namespace].add(key)
        self._records[namespace].append(
            KnowledgeRecord(
                triple=triple,
                namespace=namespace,
                source_sentence=source_sentence,
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_triples(
        self,
        namespace: str | None = None,
        subject: str | None = None,
        relation: str | None = None,
    ) -> list[Triple]:
        """Return triples matching all provided filters.

        Filters are combinable: every non-``None`` filter must match.
        Passing no filters returns all triples across all namespaces.

        Args:
            namespace: If given, restrict to this namespace.
            subject: If given, only triples where ``triple.subject == subject``.
            relation: If given, only triples where ``triple.relation == relation``.

        Returns:
            A list of ``Triple`` objects (not ``KnowledgeRecord``).
        """
        # Determine which namespaces to search
        if namespace is not None:
            namespaces = [namespace] if namespace in self._records else []
        else:
            namespaces = list(self._records.keys())

        result: list[Triple] = []
        for ns in namespaces:
            for record in self._records[ns]:
                t = record.triple
                if subject is not None and t.subject != subject:
                    continue
                if relation is not None and t.relation != relation:
                    continue
                result.append(t)

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the graph to a JSON-compatible dict.

        Format::

            {
                "namespaces": {
                    "claim": [
                        {"subject": "...", "relation": "...", "object": "...",
                         "source_sentence": "..."},
                        ...
                    ],
                    "evidence": [...]
                }
            }
        """
        namespaces: dict[str, list[dict]] = {}
        for ns, records in self._records.items():
            namespaces[ns] = [
                {
                    "subject": r.triple.subject,
                    "relation": r.triple.relation,
                    "object": r.triple.object,
                    "source_sentence": r.source_sentence,
                }
                for r in records
            ]
        return {"namespaces": namespaces}

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Reconstruct a ``KnowledgeGraph`` from the serialized dict format.

        Args:
            data: A dict as produced by :meth:`to_dict`.

        Returns:
            A fully populated ``KnowledgeGraph`` instance.
        """
        graph = cls()
        for ns, entries in data.get("namespaces", {}).items():
            for entry in entries:
                triple = Triple(
                    subject=entry["subject"],
                    relation=entry["relation"],
                    object=entry["object"],
                )
                graph.add_triple(
                    triple,
                    namespace=ns,
                    source_sentence=entry.get("source_sentence", ""),
                )
        return graph
