/**
 * MessageBubble — 聊天消息气泡，支持 Markdown 和推理追踪
 */
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../../types/chat';
import TraceViewer from '../trace/TraceViewer';

interface Props { message: Message; onTraceClick?: () => void; }

const MessageBubble: React.FC<Props> = ({ message, onTraceClick }) => {
  const [showTrace, setShowTrace] = useState(false);
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const hasTrace = !!message.trace;

  if (isSystem) {
    return <div style={{ textAlign:'center',padding:'8px',color:'var(--color-text-muted)',fontSize:'0.85em',fontStyle:'italic' }}>{message.content}</div>;
  }

  return (
    <div style={{ display:'flex',flexDirection:'column',alignItems:isUser?'flex-end':'flex-start',padding:'8px 16px',animation:'fadeIn 0.3s ease-out' }}>
      <div style={{ fontSize:'0.75em',color:'var(--color-text-muted)',marginBottom:'4px',fontWeight:600 }}>
        {isUser ? '你' : '助手'}
      </div>
      <div style={{
        maxWidth:'85%',padding:'12px 18px',
        borderRadius: isUser ? 'var(--radius-lg) var(--radius-lg) 4px var(--radius-lg)' : 'var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px',
        background: isUser ? 'var(--color-primary)' : 'var(--color-bg-secondary)',
        border: isUser ? 'none' : '1px solid var(--color-border)',
        color: isUser ? '#fff' : 'var(--color-text)', wordBreak:'break-word',
      }}>
        {isUser ? (
          <div style={{ whiteSpace:'pre-wrap' }}>{message.content}</div>
        ) : (
          <div className={`markdown-body ${message.isStreaming ? 'cursor-blink' : ''}`}>
            {message.content ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            ) : message.isStreaming ? (
              <span style={{ color:'var(--color-text-muted)',fontStyle:'italic' }}>思考中…</span>
            ) : (
              <span style={{ color:'var(--color-text-muted)' }}>（空响应）</span>
            )}
          </div>
        )}
      </div>
      {hasTrace && !isUser && (
        <button onClick={() => { setShowTrace(!showTrace); onTraceClick?.(); }}
          style={{ marginTop:'6px',padding:'4px 12px',fontSize:'0.8em',color:'var(--color-text-secondary)',background:'transparent',border:'1px solid var(--color-border)',borderRadius:'var(--radius-sm)',cursor:'pointer',transition:'all 0.2s' }}
          onMouseEnter={e => { e.currentTarget.style.background='var(--color-bg-tertiary)'; e.currentTarget.style.color='var(--color-text)' }}
          onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='var(--color-text-secondary)' }}>
          {showTrace ? '收起追踪' : '查看推理过程'} 🔍
        </button>
      )}
      {showTrace && hasTrace && message.trace && (
        <div style={{ marginTop:'8px',width:'100%',maxWidth:'85%' }}><TraceViewer trace={message.trace} /></div>
      )}
    </div>
  );
};
export default MessageBubble;
