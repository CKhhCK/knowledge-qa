/**
 * TraceViewer — Agent 推理全过程追踪面板
 */
import React from 'react';
import type { TraceInfo } from '../../types/chat';
import CategoryBadge from './CategoryBadge';
import StepTimeline from './StepTimeline';

interface Props { trace: TraceInfo; }

const TraceViewer: React.FC<Props> = ({ trace }) => (
  <div style={{ background:'var(--color-bg-secondary)',border:'1px solid var(--color-border)',borderRadius:'var(--radius-md)',padding:'16px',fontSize:'0.9em' }}>
    <div style={{ fontWeight:700,fontSize:'1.05em',marginBottom:'12px',display:'flex',alignItems:'center',gap:'8px' }}>🔍 推理追踪</div>
    <div style={{ display:'flex',flexWrap:'wrap',gap:'8px',marginBottom:'16px' }}>
      <CategoryBadge category={trace.category} confidence={trace.confidence} />
      <span style={{ padding:'3px 10px',borderRadius:'20px',background:'var(--color-bg-tertiary)',border:'1px solid var(--color-border)',fontSize:'0.85em',color:'var(--color-text-secondary)' }}>处理器: {trace.handler_used}</span>
      <span style={{ padding:'3px 10px',borderRadius:'20px',background:'var(--color-bg-tertiary)',border:'1px solid var(--color-border)',fontSize:'0.85em',color:'var(--color-text-muted)' }}>⏱️ {trace.total_duration_ms}ms</span>
    </div>
    {trace.routing_reasoning && (
      <div style={{ padding:'8px 12px',marginBottom:'12px',background:'var(--color-bg)',borderRadius:'var(--radius-sm)',fontSize:'0.85em',color:'var(--color-text-secondary)',fontStyle:'italic' }}>
        路由: {trace.routing_reasoning}
      </div>
    )}
    {trace.reflection_feedback && (
      <div style={{ padding:'10px 14px',marginBottom:'12px',background:'rgba(245,158,11,0.1)',border:'1px solid rgba(245,158,11,0.3)',borderRadius:'var(--radius-sm)' }}>
        <div style={{ fontSize:'0.8em',fontWeight:600,color:'var(--color-warning)',marginBottom:'4px' }}>
          📝 答案审核 {trace.reflection_score && <span style={{ marginLeft:'8px' }}>评分: {trace.reflection_score}/10</span>}
        </div>
        <div style={{ fontSize:'0.9em',color:'var(--color-text-secondary)',whiteSpace:'pre-wrap' }}>{trace.reflection_feedback}</div>
      </div>
    )}
    <div style={{ marginTop:'16px' }}>
      <div style={{ fontWeight:600,marginBottom:'8px',color:'var(--color-text-secondary)' }}>推理步骤 ({trace.steps.length})</div>
      <StepTimeline steps={trace.steps} />
    </div>
  </div>
);
export default TraceViewer;
