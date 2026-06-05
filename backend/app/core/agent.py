from __future__ import annotations
"""
AdvancedQAAgent — the main orchestrator for the intelligent Q&A system.

This is the central component that wires together:
- KnowledgeManager: Document ingestion, RAG pipeline, cognitive memory
- QuestionRouter: Classifies incoming questions
- Handler strategies: FactualHandler(KnowledgeBase→WebSearch),
  ReasoningHandler(ReAct), ComparisonHandler(Plan-Solve),
  CalculationHandler(CoT+Calc), MixedHandler(Decomposition)
- ReflectionVerifier: Quality gate for answers
- SessionStore: Conversation persistence

Pipeline:
  User Question
    → Classify (QuestionRouter)
    → Dispatch to handler (Strategy Pattern, with KnowledgeManager)
    → Verify quality (ReflectionVerifier)
    → Record in memory (KnowledgeManager.record_qa)
    → Store in session (SessionStore)
    → Return answer + trace
"""

import asyncio
import time
from typing import AsyncGenerator

from hello_agents import HelloAgentsLLM

from app.config import Settings, get_settings
from app.knowledge.manager import KnowledgeManager
from app.core.router import QuestionRouter, ClassificationResult
from app.core.verifier import ReflectionVerifier, VerificationResult
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.core.handlers.factual import FactualHandler
from app.core.handlers.reasoning import ReasoningHandler
from app.core.handlers.comparison import ComparisonHandler
from app.core.handlers.calculation import CalculationHandler
from app.core.handlers.mixed import MixedHandler
from app.memory.session_store import SessionStore
from app.models.chat import (
    ChatResponse,
    TraceInfo,
    StepInfo,
    ToolCallRecord,
    StreamEventModel,
)
from app.tools.registry import create_tool_registry
from app.utils.logging import get_logger
from app.utils.errors import (
    ClassificationError,
    HandlerError,
    HandlerTimeoutError,
    LLMServiceError,
)

logger = get_logger(__name__)


class AdvancedQAAgent:
    """
    Production-grade intelligent Q&A agent with multi-strategy orchestration.

    Combines agent patterns from the hello-agents curriculum:
    - Ch4: ReAct loop (reasoning handler)
    - Ch4: Plan-and-Solve (comparison handler)
    - Ch4: Reflection verification (quality gate)
    - Ch7/Ch8: Tool integration (web search, calculator, RAG, memory)
    - Ch8: Knowledge management (document ingestion, RAG pipeline)
    - Ch10: Multi-agent decomposition (mixed handler)

    Features:
    - Document ingestion & knowledge base (RAG pipeline)
    - Cognitive memory (working + episodic + semantic)
    - Automatic question classification and routing
    - Streaming SSE output for real-time response
    - Complete trace for observability
    - Session-based conversation management
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the QA agent with all components.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self.settings = settings or get_settings()

        # Initialize LLM
        self.llm = HelloAgentsLLM(
            model=self.settings.llm_model_id,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
        )
        logger.info(f"LLM initialized: {self.settings.llm_model_id}")

        # Initialize Knowledge Manager (RAG + Memory)
        self.knowledge = KnowledgeManager(self.settings, llm=self.llm)

        # Initialize tool registry with core tools
        self.tool_registry = create_tool_registry(self.settings)

        # Initialize router
        self.router = QuestionRouter(self.llm)

        # Initialize verifier
        self.verifier = ReflectionVerifier(self.llm, self.settings)

        # Initialize handlers (strategy pattern) — pass knowledge manager
        self.handlers: dict[str, HandlerInterface] = {
            "factual": FactualHandler(self.llm, self.tool_registry, self.settings, self.knowledge),
            "reasoning": ReasoningHandler(self.llm, self.tool_registry, self.settings, self.knowledge),
            "comparison": ComparisonHandler(self.llm, self.tool_registry, self.settings, self.knowledge),
            "calculation": CalculationHandler(self.llm, self.tool_registry, self.settings, self.knowledge),
        }
        # MixedHandler needs the router for recursive classification
        self.handlers["mixed"] = MixedHandler(
            self.llm, self.tool_registry, self.settings, self.router
        )

        # Session store (conversation history)
        self.session_store = SessionStore()

        logger.info(
            "AdvancedQAAgent initialized",
            extra={
                "handlers": list(self.handlers.keys()),
                "tools": self.tool_registry.list_tools(),
                "reflection_enabled": self.settings.reflection_enabled,
                "knowledge_docs": len(self.knowledge.list_documents()),
            },
        )

    async def answer(
        self,
        session_id: str,
        question: str,
        enable_reflection: bool | None = None,
        user_id: str = "anonymous",
    ) -> ChatResponse:
        """
        Process a question and return a complete answer with trace.

        Pipeline:
        1. Classify → 2. Dispatch handler (with knowledge context)
        → 3. Verify → 4. Record to memory → 5. Return

        Args:
            session_id: Session identifier for conversation continuity.
            question: The user's question text.
            enable_reflection: Override the default reflection setting.

        Returns:
            ChatResponse with answer, trace, and metadata.
        """
        start_time = time.perf_counter()

        # Ensure conversation exists
        if not self.session_store.get_conversation(session_id):
            self.session_store.create_conversation(user_id, session_id)
        logger.info(f"[REQUEST] session={session_id} | q=\"{question[:100]}\"")

        try:
            # Step 1: Classify the question
            t0 = time.perf_counter()
            classification = await self.router.classify(question)
            logger.info(
                f"[CLASSIFY] {classification.category} (conf={classification.confidence:.2f}) | "
                f"{(time.perf_counter()-t0)*1000:.0f}ms"
            )

            # Step 2: Dispatch to the appropriate handler
            handler = self.handlers.get(classification.category)
            if not handler:
                logger.warning(
                    f"Unknown category '{classification.category}', falling back to factual"
                )
                handler = self.handlers["factual"]
                classification = ClassificationResult(
                    category="factual",
                    confidence=0.3,
                    reasoning="unknown category, fallback to factual",
                )

            # Get conversation context (from session + memory)
            session_history = self.session_store.get_history(session_id)
            memory_context = self.knowledge.get_memory_context(question, user_id=user_id)
            context = {
                "history": session_history,
                "memory_context": memory_context,
            }

            t1 = time.perf_counter()
            handler_result = await handler.handle(question, context)
            logger.info(
                f"[HANDLER] {handler.handler_name} | {(time.perf_counter()-t1)*1000:.0f}ms | "
                f"steps={len(handler_result.steps)} | tool_calls={len(handler_result.tool_calls)}"
            )

            # Step 3: Verify and refine with reflection
            # Simple questions skip reflection (fast path)
            is_simple = classification.complexity == "simple"
            reflection_enabled = (
                enable_reflection
                if enable_reflection is not None
                else (self.settings.reflection_enabled and not is_simple)
            )

            if reflection_enabled:
                t2 = time.perf_counter()
                verification = await self.verifier.verify(
                    question, handler_result.answer
                )
                logger.info(
                    f"[VERIFY] score={verification.score:.1f} | refined={verification.was_refined} | "
                    f"{(time.perf_counter()-t2)*1000:.0f}ms"
                )
            else:
                verification = VerificationResult(
                    answer=handler_result.answer,
                    feedback="",
                    score=0.0,
                    was_refined=False,
                    draft_answer=handler_result.answer,
                )

            # Step 4: Record to cognitive memory
            self.knowledge.record_qa(
                question=question,
                answer=verification.answer,
                category=classification.category,
                user_id=user_id,
            )

            # Step 5: Build trace
            trace = TraceInfo(
                category=classification.category,
                confidence=classification.confidence,
                routing_reasoning=classification.reasoning,
                handler_used=handler.handler_name,
                draft_answer=verification.draft_answer,
                reflection_feedback=verification.feedback,
                reflection_score=verification.score,
                steps=handler_result.steps,
                total_duration_ms=int((time.perf_counter() - start_time) * 1000),
            )

            # Step 6: Store in session
            self.session_store.add_qa_pair(
                conversation_id=session_id,
                user_msg=question,
                assistant_msg=verification.answer,
                trace=trace,
            )

            total_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "QA pipeline completed",
                extra={
                    "session_id": session_id,
                    "category": classification.category,
                    "handler": handler.handler_name,
                    "reflection_used": reflection_enabled,
                    "reflection_score": verification.score,
                    "duration_ms": total_ms,
                },
            )

            return ChatResponse(
                session_id=session_id,
                answer=verification.answer,
                trace=trace,
            )

        except Exception as e:
            logger.error(
                "QA pipeline error",
                extra={"session_id": session_id, "error": str(e)},
                exc_info=True,
            )
            raise

    async def answer_stream(
        self,
        session_id: str,
        question: str,
        enable_reflection: bool | None = None,
        user_id: str = "anonymous",
    ) -> AsyncGenerator[str, None]:
        """
        Process a question and stream events via SSE.

        Yields SSE-formatted strings for each pipeline stage.
        See StreamEventModel for event type documentation.
        """
        start_time = time.perf_counter()
        t_log = start_time

        logger.info(f"[REQUEST] session={session_id} | q=\"{question[:100]}\"")

        def _log_step(label: str):
            nonlocal t_log
            now = time.perf_counter()
            elapsed = (now - t_log) * 1000
            total = (now - start_time) * 1000
            logger.info(f"[TIMING] {label} | step={elapsed:.0f}ms total={total:.0f}ms")
            t_log = now

        try:
            # Stage 1: Classify
            _log_step("stream_start")
            yield StreamEventModel(
                type="status", content="Analyzing question type...",
                metadata={"stage": "classifying"},
            ).model_dump_json()

            classification = await self.router.classify(question)
            _log_step(f"classified={classification.category}")

            yield StreamEventModel(
                type="status",
                content=f"Classified as {classification.category} (confidence: {classification.confidence:.0%})",
                metadata={"stage": "classified", "category": classification.category, "confidence": classification.confidence},
            ).model_dump_json()

            # Stage 2: Dispatch to handler
            handler = self.handlers.get(classification.category, self.handlers["factual"])
            yield StreamEventModel(
                type="status",
                content=f"Dispatching to {handler.handler_name} handler...",
                metadata={"stage": "dispatching", "handler": handler.handler_name},
            ).model_dump_json()

            session_history = self.session_store.get_history(session_id)
            _log_step("got_history")

            # Only search episodic memory when question has time references
            time_triggers = ["刚才", "之前", "上次", "前面", "刚刚", "那个",
                             "你提到的", "总结", "回顾", "上面", "前面说"]
            need_memory = any(t in question for t in time_triggers)
            memory_context = ""
            if need_memory:
                memory_context = self.knowledge.get_memory_context(question, user_id=user_id)
                _log_step(f"got_memory_context (triggered) len={len(memory_context)}")
            else:
                _log_step("skipped_memory_context (no trigger)")

            context = {"history": session_history, "memory_context": memory_context}

            # Check knowledge base before answering
            if self.knowledge.has_knowledge():
                yield StreamEventModel(
                    type="status",
                    content=f"Searching knowledge base ({len(self.knowledge.list_documents())} docs)...",
                    metadata={"stage": "retrieving"},
                ).model_dump_json()

            handler_result = await handler.handle(question, context)
            _log_step(f"handler_done={handler.handler_name}")

            # Report handler steps
            for step in handler_result.steps:
                if step.thought:
                    yield StreamEventModel(
                        type="thought", content=step.thought,
                        metadata={"step": step.step_number},
                    ).model_dump_json()

                if step.tool_calls:
                    for tc in step.tool_calls:
                        yield StreamEventModel(
                            type="action",
                            content=f"Calling tool: {tc.tool_name}",
                            metadata={"tool": tc.tool_name, "duration_ms": tc.duration_ms},
                        ).model_dump_json()
                        yield StreamEventModel(
                            type="observation",
                            content=tc.output_summary[:500],
                            metadata={"tool": tc.tool_name},
                        ).model_dump_json()

                if step.observation and step.step_number == len(handler_result.steps):
                    yield StreamEventModel(
                        type="text", content=handler_result.answer,
                        metadata={"step": step.step_number, "final": True},
                    ).model_dump_json()

            # Stage 3: Verify (skip for simple questions)
            is_simple = classification.complexity == "simple"
            reflection_enabled = (
                enable_reflection
                if enable_reflection is not None
                else (self.settings.reflection_enabled and not is_simple)
            )

            if reflection_enabled:
                yield StreamEventModel(
                    type="status", content="Verifying answer quality...",
                    metadata={"stage": "verifying"},
                ).model_dump_json()

                verification = await self.verifier.verify(question, handler_result.answer)
                _log_step(f"verified score={verification.score:.1f}")

                if verification.was_refined:
                    yield StreamEventModel(
                        type="status",
                        content=f"Answer refined (score: {verification.score:.0f}/10)",
                        metadata={"stage": "refined", "score": verification.score},
                    ).model_dump_json()

                final_answer = verification.answer
                final_score = verification.score
                final_feedback = verification.feedback
                was_refined = verification.was_refined
            else:
                final_answer = handler_result.answer
                final_score = 0.0
                final_feedback = ""
                was_refined = False

            # Record to cognitive memory
            self.knowledge.record_qa(
                question=question,
                answer=final_answer,
                category=classification.category,
                user_id=user_id,
            )
            _log_step("record_qa_done")

            # Build trace
            trace = TraceInfo(
                category=classification.category,
                confidence=classification.confidence,
                routing_reasoning=classification.reasoning,
                handler_used=handler.handler_name,
                draft_answer=handler_result.answer if was_refined else final_answer,
                reflection_feedback=final_feedback,
                reflection_score=final_score,
                steps=handler_result.steps,
                total_duration_ms=int((time.perf_counter() - start_time) * 1000),
            )

            # Store in session
            self.session_store.add_qa_pair(
                conversation_id=session_id,
                user_msg=question,
                assistant_msg=final_answer,
                trace=trace,
            )
            _log_step("session_stored")

            # Stage 4: Done — include FULL trace for frontend display
            total_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                f"[PIPELINE] Stream complete | session={session_id} | "
                f"category={classification.category} | handler={handler.handler_name} | "
                f"reflection_score={final_score} | duration={total_ms}ms",
                extra={"session_id": session_id, "duration_ms": total_ms, "category": classification.category},
            )

            yield StreamEventModel(
                type="done", content=final_answer,
                metadata={
                    "session_id": session_id,
                    "category": classification.category,
                    "confidence": classification.confidence,
                    "routing_reasoning": classification.reasoning,
                    "handler_used": handler.handler_name,
                    "draft_answer": handler_result.answer if was_refined else final_answer,
                    "reflection_feedback": final_feedback,
                    "reflection_score": final_score,
                    "steps": [s.model_dump() for s in handler_result.steps],
                    "total_duration_ms": total_ms,
                },
            ).model_dump_json()

        except Exception as e:
            logger.error(
                f"[PIPELINE] Stream error | session={session_id} | error={str(e)}",
                extra={"session_id": session_id, "error": str(e)},
                exc_info=True,
            )
            yield StreamEventModel(
                type="error",
                content=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
            ).model_dump_json()

    # ================================================================
    # Session & Conversation
    # ================================================================

    async def get_history(self, session_id: str) -> list[dict]:
        """Get the message history for a session."""
        return self.session_store.get_history(session_id)

    async def clear_session(self, session_id: str) -> bool:
        """Clear a session and return whether it existed."""
        return self.session_store.clear_conversation(session_id)

    # ================================================================
    # Knowledge Management
    # ================================================================

    def ingest_document(self, file_path: str, document_id: str = None,
                        chunk_strategy: str = "recursive") -> dict:
        """Ingest a document into the knowledge base."""
        return self.knowledge.ingest_document(file_path, document_id,
                                              chunk_strategy=chunk_strategy)

    def ingest_text(self, text: str, document_id: str, title: str = None,
                    chunk_strategy: str = "recursive") -> dict:
        """Ingest text content as a knowledge document."""
        return self.knowledge.ingest_text(text, document_id, title,
                                         chunk_strategy=chunk_strategy)

    def list_documents(self) -> list[dict]:
        """List all ingested knowledge documents."""
        return self.knowledge.list_documents()

    def delete_document(self, document_id: str) -> dict:
        """Delete a document from the knowledge base."""
        return self.knowledge.delete_document(document_id)

    def search_knowledge(self, query: str, limit: int = 5) -> str:
        """Semantic search over the knowledge base."""
        return self.knowledge.search_knowledge(query, limit)

    def recall_memory(self, query: str, limit: int = 5) -> str:
        """Search cognitive memory for relevant past interactions."""
        return self.knowledge.recall(query, limit)

    # ================================================================
    # System
    # ================================================================

    # ================================================================
    # Conversation Management
    # ================================================================

    def create_conversation(self, user_id: str = "anonymous") -> dict:
        """Create a new conversation."""
        return self.session_store.create_conversation(user_id)

    def list_conversations(self, user_id: str = "anonymous") -> list[dict]:
        """List all conversations for a user."""
        return self.session_store.list_conversations(user_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return self.session_store.delete_conversation(conversation_id)

    async def get_stats(self) -> dict:
        """Get comprehensive system statistics."""
        session_stats = self.session_store.get_stats()
        knowledge_stats = self.knowledge.get_stats()

        return {
            **session_stats,
            **knowledge_stats,
            "llm_model": self.settings.llm_model_id,
            "reflection_enabled": self.settings.reflection_enabled,
            "max_react_steps": self.settings.max_react_steps,
            "tools_available": self.tool_registry.list_tools(),
        }

    async def list_tools(self) -> list[str]:
        """List all available tools."""
        return self.tool_registry.list_tools()

    async def close(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("AdvancedQAAgent shutting down")
        # Consolidate important working memories before shutdown
        try:
            self.knowledge.consolidate_memories()
        except Exception as e:
            logger.warning(f"Memory consolidation failed during shutdown: {e}")
