from __future__ import annotations
"""
ReasoningHandler — handles multi-step reasoning questions using the ReAct pattern.

Implements the Thought-Action-Observation loop from Chapter 4.
The agent can use web_search and calculator tools to gather information
and perform computations before reaching a final answer.

Key references:
- code/chapter4/ReAct.py — The original ReAct pattern
- code/chapter7/my_react_agent.py — Framework-based ReActAgent
"""

import asyncio
import re
import time

from hello_agents import HelloAgentsLLM, ToolRegistry

from app.config import Settings
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.models.chat import StepInfo, ToolCallRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)

REACT_SYSTEM_PROMPT = """你是一个具备推理和行动能力的AI助手。

## 可用工具
{tools_description}

## 工作流程
请严格按照以下格式进行回应，每次回应只包含一对 Thought 和 Action：

Thought: [你的分析、推理和下一步计划]
Action: [你要执行的具体行动]

## Action 格式
1. 调用工具: `tool_name[parameters]`
   例子: `web_search[query="最新AI新闻"]`
   例子: `calculator[expression="2**10 + 5*3"]`

2. 给出最终答案: `Finish[你的完整最终答案]`

## 来源标注规则（重要）
- 如果消息中包含【知识库检索结果】，你的回答必须基于它，并在末尾标注 `📎 参考知识库文档`
- 如果知识库结果不包含答案，使用工具搜索
- 如果最终答案来自知识库 → 标注 `📎 来自知识库`
- 如果来自网络搜索 → 标注 `📎 来自网络搜索`
- 如果来自通用知识 → 标注 `⚠️ 此回答基于通用知识，知识库中未找到相关信息`

## 重要规则
1. 每次只输出一对 Thought-Action
2. 如果有知识库内容，优先基于它回答，不要重复搜索
3. Finish 必须是完整的最终答案，使用 markdown 格式

现在开始！"""


class ReasoningHandler(HandlerInterface):
    """
    Handler for multi-step reasoning questions using the ReAct pattern.

    Implements the classic Thought → Action → Observation loop:
    1. Send the question with the ReAct prompt to the LLM
    2. Parse the Thought and Action from the response
    3. Execute any tool calls and feed back Observations
    4. Repeat until Finish or max_steps reached
    """

    @property
    def handler_name(self) -> str:
        return "reasoning"

    async def handle(self, question: str, context: dict | None = None) -> HandlerResult:
        """Process a reasoning question using the ReAct loop."""
        start_time = time.perf_counter()
        steps: list[StepInfo] = []
        tool_calls: list[ToolCallRecord] = []

        # --- Pre-check knowledge base before ReAct ---
        kb_context = ""
        if self.knowledge and self.knowledge.has_knowledge():
            try:
                kb_result = await self.knowledge.ask_knowledge(
                    question, limit=3, enable_advanced=False, include_citations=True
                )
                if kb_result and len(kb_result) > 30:
                    kb_context = kb_result
                    steps.append(StepInfo(
                        step_number=0,
                        thought="先查知识库再推理",
                        action=f'rag.ask[question="{question[:60]}"]',
                        observation=kb_result[:300],
                    ))
            except Exception as e:
                logger.debug(f"Pre-check KB failed: {e}")

        # Build the prompt
        tools_desc = self.tool_registry.get_tools_description()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if kb_context:
            messages.append({
                "role": "system",
                "content": f"【知识库检索结果 — 优先基于此回答，遵守来源标注规则】:\n{kb_context}",
            })
        messages.append({"role": "user", "content": f"请解决以下问题:\n\n{question}"})

        max_steps = self.config.max_react_steps

        for step_num in range(1, max_steps + 1):
            step_start = time.perf_counter()
            logger.debug("ReAct step", extra={"step": step_num, "question": question[:80]})

            # Call LLM
            try:
                response = await asyncio.to_thread(self.llm.invoke, messages)
            except Exception as e:
                logger.error("LLM call failed in ReAct loop", extra={"error": str(e), "step": step_num})
                return HandlerResult(
                    answer=f"推理过程中出现错误: {str(e)}",
                    steps=steps,
                    tool_calls=tool_calls,
                    total_duration_ms=int((time.perf_counter() - start_time) * 1000),
                )

            # Parse thought and action
            thought, action = self._parse_response(response)

            if not action:
                # No action parsed — treat as final answer
                steps.append(StepInfo(
                    step_number=step_num,
                    thought=thought or "分析完成",
                    action="完成回答",
                    duration_ms=int((time.perf_counter() - step_start) * 1000),
                ))
                break

            logger.debug("ReAct parsed", extra={"step": step_num, "thought": (thought or "")[:100], "action": (action or "")[:100]})

            # Check for Finish
            if action.startswith("Finish"):
                final_answer = self._parse_finish(action)
                steps.append(StepInfo(
                    step_number=step_num,
                    thought=thought or "",
                    action=action[:200],
                    observation="任务完成",
                    duration_ms=int((time.perf_counter() - step_start) * 1000),
                ))
                total_ms = int((time.perf_counter() - start_time) * 1000)
                return HandlerResult(
                    answer=final_answer,
                    steps=steps,
                    tool_calls=tool_calls,
                    total_duration_ms=total_ms,
                )

            # Execute tool call
            tool_name, tool_input = self._parse_tool_action(action)
            observation = ""

            if tool_name:
                t0 = time.perf_counter()
                try:
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        result = tool.run({"input": tool_input})
                        observation = result
                        success = True
                    else:
                        observation = f"错误: 未找到工具 '{tool_name}'。可用工具: {', '.join(self.tool_registry.list_tools())}"
                        success = False
                except Exception as e:
                    observation = f"工具执行出错: {str(e)}"
                    success = False

                tool_ms = int((time.perf_counter() - t0) * 1000)
                tool_calls.append(ToolCallRecord(
                    tool_name=tool_name,
                    input_params=tool_input,
                    output_summary=observation[:500],
                    duration_ms=tool_ms,
                    success=success,
                ))
            else:
                observation = "错误: 无法解析工具调用。请使用格式: tool_name[parameters]"

            # Record step
            steps.append(StepInfo(
                step_number=step_num,
                thought=thought or "",
                action=action,
                observation=observation[:500],
                tool_calls=[tool_calls[-1]] if tool_calls else [],
                duration_ms=int((time.perf_counter() - step_start) * 1000),
            ))

            # Feed observation back into the conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # If we exhaust max_steps without Finish, generate a final answer
        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.warning("ReAct loop exhausted max steps", extra={"max_steps": max_steps, "question": question[:80]})

        # Ask LLM to synthesize from the steps
        try:
            messages.append({
                "role": "user",
                "content": "请基于以上所有推理步骤和观察结果，给出一个完整的最终答案。",
            })
            final_answer = await asyncio.to_thread(self.llm.invoke, messages)
        except Exception:
            final_answer = "抱歉，我无法在限定的推理步骤内解决这个问题。请尝试简化问题或提供更多信息。"

        return HandlerResult(
            answer=final_answer,
            steps=steps,
            tool_calls=tool_calls,
            total_duration_ms=total_ms,
        )

    def _parse_response(self, text: str) -> tuple[str | None, str | None]:
        """Parse Thought and Action from the LLM response."""
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else None

        # Extract Action
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None

        # If no structured format found, treat entire response as final
        if not thought and not action:
            return None, f"Finish[{text.strip()}]"

        return thought, action

    def _parse_tool_action(self, action: str) -> tuple[str | None, str | None]:
        """Parse tool_name[parameters] format from an action string."""
        match = re.match(r"(\w+)\s*\[(.*)\]", action, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def _parse_finish(self, action: str) -> str:
        """Parse Finish[final answer] format."""
        match = re.match(r"Finish\s*\[(.*)\]", action, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no brackets, treat the whole text after "Finish" as answer
        return action.replace("Finish", "").strip().lstrip("[").rstrip("]")
