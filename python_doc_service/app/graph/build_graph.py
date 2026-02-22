from typing import Dict, Any
import logging

from langgraph.graph import StateGraph, END

from .state import GraphState
from .nodes import (
    make_intent_classifier_node,
    make_check_feedback_node,
    make_semantic_memory_node,
    make_primary_retrieval_node,
    make_secondary_retrieval_node,
    make_tool_retrieval_node,
    make_analyzer_node,
    make_human_in_loop_node,
    make_owner_resolution_node,
    make_finalize_node
)

logger = logging.getLogger(__name__)


def build_answer_graph(embedder, call_llm, call_intent_llm=None, call_social_llm=None):
    builder = StateGraph(GraphState)

    intent_classifier = make_intent_classifier_node(call_intent_llm)
    check_feedback = make_check_feedback_node(embedder)
    semantic_memory = make_semantic_memory_node()
    primary_retrieval = make_primary_retrieval_node(embedder)
    secondary_retrieval = make_secondary_retrieval_node(embedder)
    tool_retrieval = make_tool_retrieval_node()
    analyzer = make_analyzer_node()
    human_in_loop = make_human_in_loop_node()
    owner_resolution = make_owner_resolution_node()
    finalize = make_finalize_node(call_llm, call_social_llm)

    builder.add_node("IntentClassifier", intent_classifier)
    builder.add_node("CheckFeedbackState", check_feedback)
    builder.add_node("SemanticMemoryLookup", semantic_memory)
    builder.add_node("PrimaryRetrieval", primary_retrieval)
    builder.add_node("SecondaryRetrieval", secondary_retrieval)
    builder.add_node("ToolRetrieval", tool_retrieval)
    builder.add_node("AnalyzerNode", analyzer)
    builder.add_node("HumanInLoopNode", human_in_loop)
    builder.add_node("OwnerResolutionNode", owner_resolution)
    builder.add_node("FinalizeResponse", finalize)

    builder.set_entry_point("IntentClassifier")

    def route_after_check(state: Dict[str, Any]) -> str:
        if state.get("needs_human"):
            target = "HumanInLoopNode"
        else:
            target = "SemanticMemoryLookup"
        logger.info(
            "IntentFlow route_after_check | intent=%s | needs_human=%s -> %s",
            state.get("intent_label"),
            int(bool(state.get("needs_human"))),
            target
        )
        return target

    def route_retrieval(state: Dict[str, Any]) -> str:
        if state.get("retrieval_variant") == "secondary":
            target = "SecondaryRetrieval"
        elif state.get("use_secondary_retrieval"):
            target = "SecondaryRetrieval"
        else:
            target = "PrimaryRetrieval"
        logger.info(
            "IntentFlow route_retrieval | intent=%s | retrieval_variant=%s | use_secondary=%s -> %s",
            state.get("intent_label"),
            state.get("retrieval_variant"),
            int(bool(state.get("use_secondary_retrieval"))),
            target
        )
        return target

    def route_after_analyzer(state: Dict[str, Any]) -> str:
        if state.get("needs_human") and not state.get("tool_attempted"):
            target = "ToolRetrieval"
        elif state.get("needs_human"):
            target = "HumanInLoopNode"
        elif (
            (
                state.get("low_confidence")
                or state.get("intent_label") == "latest_social_updates"
                or state.get("requires_social_search")
            ) and
            not state.get("tool_attempted")
        ):
            target = "ToolRetrieval"
        else:
            target = "OwnerResolutionNode"
        logger.info(
            "IntentFlow route_after_analyzer | intent=%s | low_confidence=%s | tool_attempted=%s | needs_human=%s | requires_social=%s -> %s",
            state.get("intent_label"),
            int(bool(state.get("low_confidence"))),
            int(bool(state.get("tool_attempted"))),
            int(bool(state.get("needs_human"))),
            int(bool(state.get("requires_social_search"))),
            target
        )
        return target

    builder.add_edge("IntentClassifier", "CheckFeedbackState")
    builder.add_conditional_edges(
        "CheckFeedbackState",
        route_after_check
    )
    builder.add_conditional_edges(
        "SemanticMemoryLookup",
        route_retrieval
    )

    builder.add_edge("PrimaryRetrieval", "AnalyzerNode")
    builder.add_edge("SecondaryRetrieval", "AnalyzerNode")
    builder.add_edge("ToolRetrieval", "AnalyzerNode")

    builder.add_conditional_edges(
        "AnalyzerNode",
        route_after_analyzer
    )

    builder.add_edge("HumanInLoopNode", "OwnerResolutionNode")
    builder.add_edge("OwnerResolutionNode", "FinalizeResponse")
    builder.add_edge("FinalizeResponse", END)

    return builder.compile()
