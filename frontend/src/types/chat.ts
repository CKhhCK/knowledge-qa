/**
 * TypeScript types mirroring the backend Pydantic models.
 * Keep these in sync with backend/app/models/chat.py
 */

export interface ChatOptions {
  enable_reflection?: boolean;
  max_react_steps?: number;
  prefer_stream?: boolean;
}

export interface ChatRequest {
  session_id: string;
  message: string;
  options?: ChatOptions;
}

export interface ToolCallRecord {
  tool_name: string;
  input_params: string;
  output_summary: string;
  duration_ms: number;
  success: boolean;
  error_message?: string;
}

export interface StepInfo {
  step_number: number;
  thought?: string;
  action?: string;
  observation?: string;
  tool_calls: ToolCallRecord[];
  timestamp: string;
  duration_ms: number;
}

export interface TraceInfo {
  category: string;
  confidence: number;
  routing_reasoning: string;
  handler_used: string;
  draft_answer?: string;
  reflection_feedback?: string;
  reflection_score?: number;
  steps: StepInfo[];
  total_duration_ms: number;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  trace?: TraceInfo;
  timestamp: string;
}

/** SSE stream event from /chat/stream */
export interface StreamEvent {
  type: 'status' | 'thought' | 'action' | 'observation' | 'text' | 'error' | 'done';
  content: string;
  metadata?: Record<string, unknown>;
}

/** UI-level message in the chat */
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  trace?: TraceInfo;
  isStreaming?: boolean;
  timestamp: string;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  preview: string;
}
