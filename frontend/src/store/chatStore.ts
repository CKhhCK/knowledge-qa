/**
 * Zustand store — chat state + multi-conversation support
 */
import { create } from 'zustand';
import type { Message, TraceInfo, StreamEvent } from '../types/chat';

export interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

function generateId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function getPersistedId(key: string): string {
  let id = localStorage.getItem(key);
  if (!id) {
    id = `conv_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem(key, id);
  }
  return id;
}

interface ChatState {
  // Current conversation
  conversationId: string;
  messages: Message[];
  currentTrace: TraceInfo | null;
  isStreaming: boolean;
  error: string | null;

  // Conversation list
  conversations: Conversation[];

  // Actions
  setConversationId: (id: string) => void;
  addUserMessage: (content: string) => string;
  startAssistantMessage: () => string;
  appendToAssistantMessage: (msgId: string, chunk: string) => void;
  finalizeAssistantMessage: (msgId: string, trace?: TraceInfo) => void;
  setTrace: (trace: TraceInfo | null) => void;
  setStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;
  clearMessages: () => void;
  handleStreamEvent: (event: StreamEvent) => void;

  // Conversation management
  setConversations: (convs: Conversation[]) => void;
  addConversation: (conv: Conversation) => void;
  removeConversation: (id: string) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversationId: getPersistedId('helloagents_qa_conv_id'),
  messages: [],
  currentTrace: null,
  isStreaming: false,
  error: null,
  conversations: [],

  setConversationId: (id) => {
    localStorage.setItem('helloagents_qa_conv_id', id);
    set({ conversationId: id });
  },

  addUserMessage: (content) => {
    const id = generateId();
    const msg: Message = { id, role: 'user', content, timestamp: new Date().toISOString() };
    set((s) => ({ messages: [...s.messages, msg] }));
    return id;
  },

  startAssistantMessage: () => {
    const id = generateId();
    const msg: Message = { id, role: 'assistant', content: '', isStreaming: true, timestamp: new Date().toISOString() };
    set((s) => ({ messages: [...s.messages, msg], isStreaming: true, error: null }));
    return id;
  },

  appendToAssistantMessage: (msgId, chunk) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === msgId ? { ...m, content: m.content + chunk } : m)),
    }));
  },

  finalizeAssistantMessage: (msgId, trace) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === msgId ? { ...m, isStreaming: false, trace } : m)),
      isStreaming: false,
      currentTrace: trace || null,
    }));
  },

  setTrace: (trace) => set({ currentTrace: trace }),
  setStreaming: (streaming) => set({ isStreaming: streaming }),
  setError: (error) => set({ error }),
  clearMessages: () => set({ messages: [], currentTrace: null }),

  handleStreamEvent: (event) => {
    const state = get();
    const assistantMsg = [...state.messages].reverse().find((m) => m.role === 'assistant' && m.isStreaming);
    switch (event.type) {
      case 'text':
        if (assistantMsg) get().appendToAssistantMessage(assistantMsg.id, event.content);
        break;
      case 'done':
        if (assistantMsg) {
          get().finalizeAssistantMessage(assistantMsg.id, (event.metadata as any) ?? undefined);
          if (event.content) {
            set((s) => ({ messages: s.messages.map((m) => (m.id === assistantMsg.id ? { ...m, content: event.content } : m)) }));
          }
        }
        break;
      case 'error':
        set({ error: event.content, isStreaming: false });
        if (assistantMsg) get().finalizeAssistantMessage(assistantMsg.id);
        break;
    }
  },

  setConversations: (convs) => set({ conversations: convs }),
  addConversation: (conv) => set((s) => ({ conversations: [conv, ...s.conversations] })),
  removeConversation: (id) => set((s) => ({ conversations: s.conversations.filter((c) => c.conversation_id !== id) })),
}));
