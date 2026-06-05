/**
 * ToolCallCard — 工具调用详情卡片，可展开
 */
import React, { useState } from 'react';
import type { ToolCallRecord } from '../../types/chat';

interface Props { call: ToolCallRecord; }

const ToolCallCard: React.FC<Props> = ({ call }) => {
  const [on, setOn] = useState(false);
  return (
    <div style={{ background:'var(--color-bg)',borderRadius:'var(--radius-sm)',border:'1px solid var(--color-border)',padding:'10px 14px',marginTop:'8px',fontSize:'0.85em' }}>
      <div onClick={() => setOn(!on)} style={{ display:'flex',justifyContent:'space-between',alignItems:'center',cursor:'pointer',userSelect:'none' }}>
        <div style={{ display:'flex',alignItems:'center',gap:'8px' }}>
          <span style={{ width:'6px',height:'6px',borderRadius:'50%',background:call.success?'var(--color-success)':'var(--color-error)' }} />
          <span style={{ fontWeight:600,color:'var(--color-primary-hover)' }}>🔧 {call.tool_name}</span>
        </div>
        <span style={{ color:'var(--color-text-muted)',fontSize:'0.85em' }}>{call.duration_ms}ms {on?'▲':'▼'}</span>
      </div>
      {on && (
        <div style={{ marginTop:'10px' }} className="animate-fade-in">
          <div style={{ marginBottom:'8px' }}>
            <div style={{ fontSize:'0.8em',color:'var(--color-text-muted)',marginBottom:'2px',fontWeight:600 }}>输入:</div>
            <code style={{ display:'block',padding:'6px 10px',background:'var(--color-bg-tertiary)',borderRadius:'4px',fontSize:'0.9em',wordBreak:'break-all' }}>{call.input_params}</code>
          </div>
          <div>
            <div style={{ fontSize:'0.8em',color:'var(--color-text-muted)',marginBottom:'2px',fontWeight:600 }}>输出:</div>
            <div style={{ padding:'6px 10px',background:'var(--color-bg-tertiary)',borderRadius:'4px',fontSize:'0.9em',maxHeight:'200px',overflowY:'auto',whiteSpace:'pre-wrap',wordBreak:'break-word' }}>{call.output_summary}</div>
          </div>
        </div>
      )}
    </div>
  );
};
export default ToolCallCard;
