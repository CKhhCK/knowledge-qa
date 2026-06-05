/**
 * CategoryBadge — 问题分类彩色标签
 */
import React from 'react';

const COLORS: Record<string, string> = {
  factual:'var(--color-factual)', reasoning:'var(--color-reasoning)',
  comparison:'var(--color-comparison)', calculation:'var(--color-calculation)',
  mixed:'var(--color-mixed)',
};
const LABELS: Record<string, string> = {
  factual:'知识问答', reasoning:'推理分析', comparison:'对比分析',
  calculation:'数学计算', mixed:'复合问题',
};

interface Props { category: string; confidence?: number; }

const CategoryBadge: React.FC<Props> = ({ category, confidence }) => {
  const c = COLORS[category] || 'var(--color-text-muted)';
  const l = LABELS[category] || category;
  return (
    <span style={{ display:'inline-flex',alignItems:'center',gap:'6px',padding:'3px 10px',borderRadius:'20px',background:`${c}20`,border:`1px solid ${c}40`,color:c,fontSize:'0.8em',fontWeight:600 }}>
      <span style={{ width:'8px',height:'8px',borderRadius:'50%',background:c }} />
      {l}
      {confidence !== undefined && <span style={{ opacity:0.8,fontSize:'0.9em' }}>{Math.round(confidence*100)}%</span>}
    </span>
  );
};
export default CategoryBadge;
