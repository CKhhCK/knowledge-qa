/**
 * ChatContainer — Main chat layout component.
 *
 * Composes:
 * - Message list (scrollable)
 * - MessageInput (fixed bottom)
 * - Empty state (when no messages)
 * - Error banner
 */

import React, { useRef, useEffect } from 'react';
import { useChatStore } from '../../store/chatStore';
import { useChat } from '../../hooks/useChat';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';
import EmptyState from './EmptyState';

const ChatContainer: React.FC = () => {
  const messages = useChatStore((s) => s.messages);
  const error = useChatStore((s) => s.error);
  const currentTrace = useChatStore((s) => s.currentTrace);
  const setError = useChatStore((s) => s.setError);
  const { send, isStreaming, loadHistory } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load conversation history when page mounts
  useEffect(() => { loadHistory(); }, [loadHistory]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--color-bg)',
    }}>
      {/* Error banner */}
      {error && (
        <div style={{
          padding: '10px 16px',
          background: 'rgba(239, 68, 68, 0.15)',
          borderBottom: '1px solid var(--color-error)',
          color: 'var(--color-error)',
          fontSize: '0.9em',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>⚠️ {error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-error)',
              cursor: 'pointer',
              fontSize: '1.2em',
              padding: '0 4px',
            }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Messages area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px 0',
      }}>
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <MessageInput onSend={send} isStreaming={isStreaming} />
    </div>
  );
};

export default ChatContainer;
