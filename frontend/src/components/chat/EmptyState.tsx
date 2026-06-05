/**
 * EmptyState — 首次进入时展示的欢迎页和示例问题
 */
import React from 'react';
import { useChat } from '../../hooks/useChat';

const EXAMPLE_QUESTIONS = [
  { icon: '📖', label: '知识问答', question: '什么是大语言模型（LLM）？', color: 'var(--color-factual)' },
  { icon: '🧠', label: '推理分析', question: '为什么Transformer架构比RNN更有效？', color: 'var(--color-reasoning)' },
  { icon: '⚖️', label: '对比分析', question: 'Python和Go在后端开发上有什么区别？', color: 'var(--color-comparison)' },
  { icon: '🔢', label: '数学计算', question: '2的10次方加上100等于多少？', color: 'var(--color-calculation)' },
  { icon: '🔀', label: '复合问题', question: '什么是RAG？它有什么优缺点？如何实现？', color: 'var(--color-mixed)' },
];

const EmptyState: React.FC = () => {
  const { send } = useChat();
  return (
    <div style={{ display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',minHeight:'60vh',padding:'32px',textAlign:'center' }}>
      <div style={{ fontSize:'3em',marginBottom:'16px' }}>🤖</div>
      <h1 style={{ fontSize:'1.8em',fontWeight:700,marginBottom:'8px',color:'var(--color-text)' }}>HelloAgents 智能问答</h1>
      <p style={{ fontSize:'1.05em',color:'var(--color-text-secondary)',maxWidth:'500px',marginBottom:'32px',lineHeight:1.6 }}>
        基于多策略 AI Agent 的智能问答系统。自动识别问题类型，选择最优处理策略，审核答案质量。
      </p>
      <div style={{ display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))',gap:'12px',maxWidth:'700px',width:'100%' }}>
        {EXAMPLE_QUESTIONS.map((ex) => (
          <button key={ex.label} onClick={() => send(ex.question)}
            style={{ padding:'14px 18px',borderRadius:'var(--radius-md)',border:'1px solid var(--color-border)',background:'var(--color-bg-secondary)',color:'var(--color-text)',fontSize:'0.9em',textAlign:'left',cursor:'pointer',transition:'all 0.2s',display:'flex',alignItems:'flex-start',gap:'10px' }}
            onMouseEnter={e => { e.currentTarget.style.borderColor=ex.color; e.currentTarget.style.background='var(--color-bg-tertiary)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor='var(--color-border)'; e.currentTarget.style.background='var(--color-bg-secondary)' }}>
            <span style={{ fontSize:'1.3em',flexShrink:0 }}>{ex.icon}</span>
            <div>
              <div style={{ fontSize:'0.75em',fontWeight:600,color:ex.color,marginBottom:'2px' }}>{ex.label}</div>
              <div style={{ lineHeight:1.5 }}>{ex.question}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
export default EmptyState;
