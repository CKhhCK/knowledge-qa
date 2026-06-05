/**
 * HistorySidebar — 会话列表侧边栏，支持多会话切换
 */
import React, { useEffect } from 'react';
import { useChatStore, Conversation } from '../../store/chatStore';

const HistorySidebar: React.FC = () => {
  const conversations = useChatStore((s) => s.conversations);
  const currentId = useChatStore((s) => s.conversationId);
  const setConversationId = useChatStore((s) => s.setConversationId);
  const setConversations = useChatStore((s) => s.setConversations);
  const clearMessages = useChatStore((s) => s.clearMessages);

  // Load conversations on mount
  useEffect(() => { loadConversations(); }, []);

  const loadConversations = async () => {
    try {
      const res = await fetch('/api/v1/chat/conversations');
      const data = await res.json();
      if (data.conversations) setConversations(data.conversations);
    } catch {}
  };

  const handleNewChat = async () => {
    try {
      const res = await fetch('/api/v1/chat/conversations', { method: 'POST' });
      const conv = await res.json();
      setConversationId(conv.conversation_id);
      clearMessages();
      loadConversations();
    } catch {}
  };

  const handleSwitch = async (conv: Conversation) => {
    setConversationId(conv.conversation_id);
    clearMessages();
    // Load history for selected conversation
    try {
      const res = await fetch(`/api/v1/chat/${conv.conversation_id}/history`);
      const data = await res.json();
      if (data.messages) {
        for (const msg of data.messages) {
          if (msg.role === 'user') {
            useChatStore.getState().addUserMessage(msg.content);
          } else if (msg.role === 'assistant' && msg.content) {
            const id = useChatStore.getState().startAssistantMessage();
            useChatStore.getState().appendToAssistantMessage(id, msg.content);
            useChatStore.getState().finalizeAssistantMessage(id, msg.trace ?? undefined);
          }
        }
      }
    } catch {}
  };

  const handleDelete = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`/api/v1/chat/conversations/${convId}`, { method: 'DELETE' });
      useChatStore.getState().removeConversation(convId);
      if (convId === currentId) {
        // Create a new one if we deleted current
        handleNewChat();
      }
    } catch {}
  };

  return (
    <div style={{
      width: '260px', height: '100%', background: 'var(--color-bg-secondary)',
      borderRight: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        padding: '16px', borderBottom: '1px solid var(--color-border)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontWeight: 600, fontSize: '0.95em', color: 'var(--color-text)' }}>会话列表</span>
        <button
          onClick={handleNewChat}
          style={{
            padding: '6px 14px', borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--color-border)', background: 'var(--color-primary)',
            color: '#fff', cursor: 'pointer', fontSize: '0.85em', fontWeight: 600,
          }}
        >+ 新建</button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
        {conversations.length === 0 && (
          <div style={{ padding: '16px', color: 'var(--color-text-muted)', fontSize: '0.85em', textAlign: 'center' }}>
            暂无会话，点击"新建"开始
          </div>
        )}
        {conversations.map((conv) => (
          <div
            key={conv.conversation_id}
            onClick={() => handleSwitch(conv)}
            style={{
              padding: '12px', marginBottom: '4px', borderRadius: 'var(--radius-sm)',
              cursor: 'pointer', transition: 'all 0.15s',
              background: conv.conversation_id === currentId ? 'var(--color-bg-tertiary)' : 'transparent',
              border: conv.conversation_id === currentId ? '1px solid var(--color-border)' : '1px solid transparent',
            }}
            onMouseEnter={(e) => {
              if (conv.conversation_id !== currentId)
                e.currentTarget.style.background = 'var(--color-bg-tertiary)';
            }}
            onMouseLeave={(e) => {
              if (conv.conversation_id !== currentId)
                e.currentTarget.style.background = 'transparent';
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: '0.85em', fontWeight: 500, color: 'var(--color-text)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {conv.title || '新对话'}
                </div>
                <div style={{ fontSize: '0.75em', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                  {conv.message_count} 条消息
                </div>
              </div>
              <button
                onClick={(e) => handleDelete(conv.conversation_id, e)}
                title="删除"
                style={{
                  background: 'none', border: 'none', color: 'var(--color-text-muted)',
                  cursor: 'pointer', fontSize: '0.85em', padding: '2px 6px',
                  opacity: 0.6, flexShrink: 0, marginLeft: '8px',
                }}
              >✕</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistorySidebar;
