/**
 * MessageInput — 消息输入框。Enter 发送，Shift+Enter 换行
 */
import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';

interface Props { onSend: (msg: string) => void; isStreaming: boolean; disabled?: boolean; }

const MessageInput: React.FC<Props> = ({ onSend, isStreaming, disabled }) => {
  const [value, setValue] = useState('');
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = taRef.current;
    if (ta) { ta.style.height = 'auto'; ta.style.height = `${Math.min(ta.scrollHeight, 150)}px`; }
  }, [value]);
  useEffect(() => { taRef.current?.focus(); }, []);

  const send = () => {
    const t = value.trim();
    if (!t || isStreaming) return;
    onSend(t); setValue('');
    if (taRef.current) taRef.current.style.height = 'auto';
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div style={{ padding:'16px',background:'var(--color-bg-secondary)',borderTop:'1px solid var(--color-border)' }}>
      <div style={{ display:'flex',gap:'12px',alignItems:'flex-end',maxWidth:'900px',margin:'0 auto' }}>
        <textarea ref={taRef} value={value} onChange={e => setValue(e.target.value)} onKeyDown={onKey}
          placeholder="输入你的问题…（Enter 发送，Shift+Enter 换行）"
          disabled={disabled || isStreaming} rows={1}
          style={{ flex:1,padding:'12px 16px',borderRadius:'var(--radius-md)',border:'1px solid var(--color-border)',background:'var(--color-bg)',color:'var(--color-text)',fontSize:'0.95em',resize:'none',outline:'none',fontFamily:'inherit',lineHeight:1.5,transition:'border-color 0.2s' }}
          onFocus={e => e.currentTarget.style.borderColor='var(--color-primary)'}
          onBlur={e => e.currentTarget.style.borderColor='var(--color-border)'} />
        <button onClick={send} disabled={!value.trim() || isStreaming || disabled}
          style={{ padding:'12px 24px',borderRadius:'var(--radius-md)',border:'none',background:isStreaming||!value.trim()?'var(--color-bg-tertiary)':'var(--color-primary)',color:isStreaming||!value.trim()?'var(--color-text-muted)':'#fff',fontSize:'0.95em',fontWeight:600,cursor:isStreaming||!value.trim()?'not-allowed':'pointer',transition:'all 0.2s',whiteSpace:'nowrap' }}>
          {isStreaming ? '…' : '发送 →'}
        </button>
      </div>
      <div style={{ display:'flex',justifyContent:'space-between',maxWidth:'900px',margin:'6px auto 0',fontSize:'0.75em',color:'var(--color-text-muted)' }}>
        <span>Enter 发送 · Shift+Enter 换行</span>
        <span>{value.length} / 4000</span>
      </div>
    </div>
  );
};
export default MessageInput;
