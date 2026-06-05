/**
 * RegisterPage — 用户注册页面
 */
import React, { useState } from 'react';
import { useAuthStore } from '../store/authStore';

const API = '/api/v1';

interface Props { onRegister: () => void; onSwitchToLogin: () => void; }

const RegisterPage: React.FC<Props> = ({ onRegister, onSwitchToLogin }) => {
  const setAuth = useAuthStore(s => s.setAuth);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (username.length < 2) { setError('用户名至少2个字符'); return; }
    if (password.length < 6) { setError('密码至少6个字符'); return; }
    if (password !== password2) { setError('两次密码不一致'); return; }

    setLoading(true);
    try {
      const res = await fetch(`${API}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '注册失败');

      setAuth(data.access_token, {
        user_id: data.user_id,
        username: data.username,
      });
      onRegister();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',minHeight:'100vh',background:'var(--color-bg)' }}>
      <div style={{ width:'380px',padding:'40px',background:'var(--color-bg-secondary)',borderRadius:'var(--radius-lg)',border:'1px solid var(--color-border)' }}>
        <div style={{ textAlign:'center',marginBottom:'32px' }}>
          <div style={{ fontSize:'3em',marginBottom:'8px' }}>🤖</div>
          <h1 style={{ fontSize:'1.5em',fontWeight:700,color:'var(--color-text)' }}>创建账号</h1>
          <p style={{ color:'var(--color-text-secondary)',fontSize:'0.9em',marginTop:'8px' }}>注册后即可使用智能问答</p>
        </div>

        {error && (
          <div style={{ padding:'10px 14px',marginBottom:'16px',background:'rgba(239,68,68,0.1)',border:'1px solid var(--color-error)',borderRadius:'var(--radius-sm)',color:'var(--color-error)',fontSize:'0.85em' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom:'16px' }}>
            <label style={{ display:'block',fontSize:'0.85em',fontWeight:600,color:'var(--color-text)',marginBottom:'6px' }}>用户名</label>
            <input type="text" value={username} onChange={e => setUsername(e.target.value)}
              placeholder="2-32位字母、数字或下划线"
              style={{ width:'100%',padding:'12px 14px',borderRadius:'var(--radius-md)',border:'1px solid var(--color-border)',background:'var(--color-bg)',color:'var(--color-text)',fontSize:'0.95em',outline:'none',boxSizing:'border-box' }}
              autoFocus />
          </div>

          <div style={{ marginBottom:'16px' }}>
            <label style={{ display:'block',fontSize:'0.85em',fontWeight:600,color:'var(--color-text)',marginBottom:'6px' }}>密码</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)}
              placeholder="至少6位密码"
              style={{ width:'100%',padding:'12px 14px',borderRadius:'var(--radius-md)',border:'1px solid var(--color-border)',background:'var(--color-bg)',color:'var(--color-text)',fontSize:'0.95em',outline:'none',boxSizing:'border-box' }} />
          </div>

          <div style={{ marginBottom:'24px' }}>
            <label style={{ display:'block',fontSize:'0.85em',fontWeight:600,color:'var(--color-text)',marginBottom:'6px' }}>确认密码</label>
            <input type="password" value={password2} onChange={e => setPassword2(e.target.value)}
              placeholder="再次输入密码"
              style={{ width:'100%',padding:'12px 14px',borderRadius:'var(--radius-md)',border:'1px solid var(--color-border)',background:'var(--color-bg)',color:'var(--color-text)',fontSize:'0.95em',outline:'none',boxSizing:'border-box' }} />
          </div>

          <button type="submit" disabled={loading || !username || !password}
            style={{ width:'100%',padding:'12px',borderRadius:'var(--radius-md)',border:'none',background:loading?'var(--color-bg-tertiary)':'var(--color-primary)',color:loading?'var(--color-text-muted)':'#fff',fontSize:'1em',fontWeight:600,cursor:loading?'not-allowed':'pointer',transition:'all 0.2s' }}>
            {loading ? '注册中…' : '注册'}
          </button>
        </form>

        <div style={{ textAlign:'center',marginTop:'20px',fontSize:'0.85em',color:'var(--color-text-secondary)' }}>
          已有账号？{' '}
          <button onClick={onSwitchToLogin}
            style={{ background:'none',border:'none',color:'var(--color-primary-hover)',cursor:'pointer',fontSize:'inherit',fontWeight:600 }}>
            去登录
          </button>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;
