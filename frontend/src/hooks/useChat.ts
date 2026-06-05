/**
 * Hook for sending messages and managing streaming responses.
 */
import { useCallback, useRef, useEffect } from 'react';
import { useChatStore } from '../store/chatStore';
import { streamMessage, sendMessage, ApiError } from '../services/api';
import type { ChatRequest } from '../types/chat';

export function useChat() {
  const store = useChatStore();
  const historyLoaded = useRef(false);

  const send = useCallback(async (message: string) => {
    if (!message.trim() || store.isStreaming) return;
    store.addUserMessage(message);
    const assistantId = store.startAssistantMessage();
    const request: ChatRequest = {
      session_id: store.conversationId,
      message: message.trim(),
      options: { prefer_stream: true },
    };
    await streamMessage(
      request,
      (event) => { store.handleStreamEvent(event); },
      (error) => {
        const msg = error instanceof ApiError
          ? `请求失败 (${error.status}): ${error.detail}`
          : `连接错误: ${error.message}`;
        store.setError(msg);
        store.finalizeAssistantMessage(assistantId);
      },
      () => {},
    );
  }, [store.conversationId, store.isStreaming]);

  const sendNonStreaming = useCallback(async (message: string) => {
    if (!message.trim() || store.isStreaming) return;
    store.addUserMessage(message);
    const assistantId = store.startAssistantMessage();
    try {
      const response = await sendMessage({ session_id: store.conversationId, message: message.trim() });
      store.appendToAssistantMessage(assistantId, response.answer);
      store.finalizeAssistantMessage(assistantId, response.trace || undefined);
    } catch (error) {
      store.setError(error instanceof ApiError ? `请求失败 (${error.status})` : `请求错误: ${String(error)}`);
      store.finalizeAssistantMessage(assistantId);
    }
  }, [store.conversationId, store.isStreaming]);

  const clearChat = useCallback(async () => {
    try {
      const { clearSession } = await import('../services/api');
      await clearSession(store.conversationId);
    } catch {}
    store.clearMessages();
  }, [store.conversationId]);

  /** Load history ONCE on mount — retry on connection failure */
  const loadHistory = useCallback(async () => {
    if (historyLoaded.current) return;
    historyLoaded.current = true;

    // Retry up to 3 times with delay (backend may still be starting)
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const { getHistory } = await import('../services/api');
        const result = await getHistory(store.conversationId);
        if (result.messages && result.messages.length > 0) {
          store.clearMessages();
          for (const msg of result.messages) {
            if (msg.role === 'user') {
              store.addUserMessage(msg.content);
            } else if (msg.role === 'assistant' && msg.content) {
              const id = store.startAssistantMessage();
              store.appendToAssistantMessage(id, msg.content);
              store.finalizeAssistantMessage(id, (msg as any).trace ?? undefined);
            }
          }
        }
        return; // Success, stop retrying
      } catch {
        if (attempt < 2) await new Promise(r => setTimeout(r, 2000)); // Wait 2s before retry
      }
    }
  }, [store.conversationId]);

  return {
    send, sendNonStreaming, clearChat, loadHistory,
    isStreaming: store.isStreaming, error: store.error,
  };
}
