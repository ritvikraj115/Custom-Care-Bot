from typing import TypedDict, Any, List, Dict


class GraphState(TypedDict, total=False):
    # Request
    query: str
    bot_id: str
    client_id: str
    retrieval_variant: str
    doc_scope: str | None
    top_k: int
    exclude_chunk_refs: list[str] | None
    conversation: List[Dict[str, Any]] | None
    trace: List[Dict[str, Any]]
    followup_context: str | None
    followup_intent: bool
    no_docs: bool

    # Intent classification
    intent_label: str
    intent_reason: str
    intent_confidence: float
    use_secondary_retrieval: bool
    requires_social_search: bool
    prefer_memory: bool
    needs_clarification_intent: bool
    social_links: Dict[str, str] | None
    website_url: str | None

    # Memory
    query_embedding: Any
    experience_hit: Dict[str, Any] | None

    # Retrieval
    chunks: List[Dict[str, Any]]
    context: str
    references: List[Dict[str, Any]]
    confidence: float
    source_type: str
    answer: str

    # Tooling / flow flags
    tool_results: Dict[str, Any]
    tool_attempted: bool
    tool_has_results: bool
    needs_human: bool
    regen_failed: bool
    is_ambiguous: bool
    low_confidence: bool
    feedback_block: bool

    # Owner override (future)
    owner_override: Dict[str, Any] | None

    # Error
    error: str | None
