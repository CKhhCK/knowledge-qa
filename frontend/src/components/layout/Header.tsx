/**
 * Header — 顶部标题栏，含用户名和退出按钮
 */
import React from 'react';
import { useAuthStore } from '../../store/authStore';

interface Props { onToggleSettings: () => void; }

const Header: React.FC<Props> = ({ onToggleSettings }) => {
  const user = useAuthStore(s => s.user);
  const logout = useAuthStore(s => s.logout);

  return (
    <header style={{ display:'flex',justifyContent:'space-between',alignItems:'center',padding:'12px 20px',background:'var(--color-bg-secondary)',borderBottom:'1px solid var(--color-border)' }}>
      <div style={{ display:'flex',alignItems:'center',gap:'10px' }}>
        <span style={{ fontSize:'1.4em' }}>🤖</span>
        <div>
          <h1 style={{ fontSize:'1.1em',fontWeight:700,color:'var(--color-text)' }}>HelloAgents 智能问答</h1>
          <p style={{ fontSize:'0.75em',color:'var(--color-text-muted)' }}>多策略 AI Agent 问答系统</p>
        </div>
      </div>

      <div style={{ display:'flex',alignItems:'center',gap:'12px' }}>
        <span style={{ fontSize:'0.85em',color:'var(--color-text-secondary)' }}>
          👤 {user?.username || ''}
        </span>
        <button onClick={logout}
          style={{ padding:'6px 14px',fontSize:'0.8em',color:'var(--color-text-secondary)',background:'transparent',border:'1px solid var(--color-border)',borderRadius:'var(--radius-sm)',cursor:'pointer',transition:'all 0.2s' }}
          onMouseEnter={e => { e.currentTarget.style.background='var(--color-bg-tertiary)'; e.currentTarget.style.color='var(--color-error)' }}
          onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='var(--color-text-secondary)' }}>
          退出
        </button>
        <button onClick={onToggleSettings} title="设置"
          style={{ width:'36px',height:'36px',borderRadius:'var(--radius-sm)',border:'1px solid var(--color-border)',background:'var(--color-bg)',color:'var(--color-text-secondary)',fontSize:'1.1em',cursor:'pointer',display:'flex',alignItems:'center',justifyContent:'center',transition:'all 0.2s' }}
          onMouseEnter={e => { e.currentTarget.style.background='var(--color-bg-tertiary)'; e.currentTarget.style.color='var(--color-text)' }}
          onMouseLeave={e => { e.currentTarget.style.background='var(--color-bg)'; e.currentTarget.style.color='var(--color-text-secondary)' }}>
          ⚙️
        </button>
      </div>
    </header>
  );
};
export default Header;
