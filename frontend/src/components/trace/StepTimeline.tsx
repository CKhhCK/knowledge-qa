/**
 * StepTimeline — Agent 推理步骤垂直时间线
 */
import React from 'react';
import type { StepInfo } from '../../types/chat';
import ToolCallCard from './ToolCallCard';

interface Props { steps: StepInfo[]; }

const StepTimeline: React.FC<Props> = ({ steps }) => {
  if (!steps?.length) return null;
  return (
    <div style={{ position:'relative' }}>
      <div style={{ position:'absolute',left:'15px',top:'12px',bottom:'12px',width:'2px',background:'var(--color-border)' }} />
      {steps.map(step => (
        <div key={step.step_number} style={{ position:'relative',paddingLeft:'44px',marginBottom:'20px' }}>
          <div style={{ position:'absolute',left:'8px',top:'2px',width:'16px',height:'16px',borderRadius:'50%',background:'var(--color-primary)',border:'3px solid var(--color-bg-secondary)',zIndex:1 }} />
          <div style={{ fontSize:'0.85em' }}>
            <div style={{ display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'4px' }}>
              <span style={{ fontWeight:600,color:'var(--color-primary-hover)' }}>步骤 {step.step_number}</span>
              <span style={{ fontSize:'0.85em',color:'var(--color-text-muted)' }}>{step.duration_ms}ms</span>
            </div>
            {step.thought && (
              <div style={{ padding:'8px 12px',background:'var(--color-bg-tertiary)',borderRadius:'var(--radius-sm)',marginBottom:'6px',borderLeft:'3px solid var(--color-info)' }}>
                <div style={{ fontSize:'0.8em',color:'var(--color-text-muted)',marginBottom:'2px',fontWeight:600 }}>💭 思考</div>
                <div>{step.thought}</div>
              </div>
            )}
            {step.action && (
              <div style={{ padding:'8px 12px',background:'var(--color-bg-tertiary)',borderRadius:'var(--radius-sm)',marginBottom:'6px',borderLeft:'3px solid var(--color-warning)' }}>
                <div style={{ fontSize:'0.8em',color:'var(--color-text-muted)',marginBottom:'2px',fontWeight:600 }}>🎬 行动</div>
                <code style={{ fontSize:'0.9em' }}>{step.action}</code>
              </div>
            )}
            {step.observation && (
              <div style={{ padding:'8px 12px',background:'var(--color-bg-tertiary)',borderRadius:'var(--radius-sm)',borderLeft:'3px solid var(--color-success)' }}>
                <div style={{ fontSize:'0.8em',color:'var(--color-text-muted)',marginBottom:'2px',fontWeight:600 }}>👁️ 观察</div>
                <div style={{ maxHeight:'150px',overflowY:'auto',whiteSpace:'pre-wrap' }}>{step.observation}</div>
              </div>
            )}
            {step.tool_calls.map((call, idx) => <ToolCallCard key={idx} call={call} />)}
          </div>
        </div>
      ))}
    </div>
  );
};
export default StepTimeline;
