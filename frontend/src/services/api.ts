/**
 * API client for the HelloAgents QA backend.
 *
 * Uses fetch with proper error handling.
 * Base URL is configurable via VITE_API_BASE_URL env var.
 */

import type { ChatRequest, ChatResponse, SessionInfo, StreamEvent } from '../types/chat';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public errorType?: string,
  ) {
    super(detail);
    this.name = 'ApiError';
  }
}

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('helloagents_qa_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers as Record<string, string>,
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorBody = await response.json();
      detail = errorBody.detail || detail;
    } catch {
      // Could not parse error body
    }
    throw new ApiError(response.status, detail);
  }

  return response.json();
}

// --- Chat API ---

export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function streamMessage(
  req: ChatRequest,
  onEvent: (event: StreamEvent) => void,
  onError: (error: Error) => void,
  onComplete: () => void,
): Promise<void> {
  const url = `${BASE_URL}/chat/stream`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
      body: JSON.stringify(req),
    });

    if (!response.ok) {
      throw new ApiError(response.status, `Stream request failed: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE messages: "event: message\ndata: {...}\n\n"
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      let eventType = '';
      let eventData = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          eventData = line.slice(6).trim();
          if (eventType && eventData) {
            try {
              const event = JSON.parse(eventData) as StreamEvent;
              onEvent(event);
            } catch {
              console.warn('Failed to parse SSE event:', eventData);
            }
            eventType = '';
            eventData = '';
          }
        }
      }
    }

    onComplete();
  } catch (error) {
    onError(error instanceof Error ? error : new Error(String(error)));
  }
}

// --- Session API ---

export async function getHistory(sessionId: string): Promise<{
  session_id: string;
  messages: Array<{ role: string; content: string; trace?: unknown; timestamp: string }>;
  message_count: number;
}> {
  return request(`/chat/${sessionId}/history`);
}

export async function listSessions(): Promise<{
  sessions: SessionInfo[];
  count: number;
}> {
  return request('/chat/sessions');
}

export async function clearSession(sessionId: string): Promise<{ status: string }> {
  return request(`/chat/${sessionId}`, { method: 'DELETE' });
}

// --- Health API ---

export async function healthCheck(): Promise<{ status: string; service: string }> {
  return request('/health');
}

export async function getStats(): Promise<Record<string, unknown>> {
  return request('/admin/stats');
}

export { ApiError };
