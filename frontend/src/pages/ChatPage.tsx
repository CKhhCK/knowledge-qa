/**
 * ChatPage — 主页面：会话侧边栏 + 聊天区 + 设置
 */
import React, { useState } from 'react';
import Header from '../components/layout/Header';
import ChatContainer from '../components/chat/ChatContainer';
import Sidebar from '../components/layout/Sidebar';

const ChatPage: React.FC = () => {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div style={{ display:'flex',flexDirection:'column',height:'100vh',overflow:'hidden' }}>
      <Header onToggleSettings={() => setShowSettings(!showSettings)} />
      <div style={{ flex:1,display:'flex',overflow:'hidden' }}>
        {/* 侧边栏：会话 + 知识库 */}
        <Sidebar />
        {/* 聊天区 */}
        <div style={{ flex:1,display:'flex',flexDirection:'column' }}>
          <ChatContainer />
        </div>
        {/* 设置 */}
        {showSettings && (
          <aside style={{ width:'300px',background:'var(--color-bg-secondary)',borderLeft:'1px solid var(--color-border)',padding:'20px',overflowY:'auto' }}>
            <div style={{ display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px' }}>
              <h2 style={{ fontSize:'1em',fontWeight:600 }}>⚙️ 设置</h2>
              <button onClick={() => setShowSettings(false)} style={{ background:'none',border:'none',color:'var(--color-text-muted)',cursor:'pointer',fontSize:'1.2em' }}>✕</button>
            </div>
            <div style={{ fontSize:'0.85em',color:'var(--color-text-secondary)' }}>
              <h3 style={{ fontSize:'0.9em',marginBottom:'8px',color:'var(--color-text)' }}>关于</h3>
              <p style={{ lineHeight:1.6,marginBottom:'16px' }}>HelloAgents QA 使用多策略 AI Agent 来回答问题。</p>
              <h3 style={{ fontSize:'0.9em',marginBottom:'8px',color:'var(--color-text)' }}>Agent 策略</h3>
              <ul style={{ paddingLeft:'16px',lineHeight:1.8 }}>
                <li><strong style={{ color:'var(--color-factual)' }}>知识问答</strong> — 知识库检索</li>
                <li><strong style={{ color:'var(--color-reasoning)' }}>推理分析</strong> — 多步 ReAct 循环</li>
                <li><strong style={{ color:'var(--color-comparison)' }}>对比分析</strong> — 先规划再执行</li>
                <li><strong style={{ color:'var(--color-calculation)' }}>数学计算</strong> — 安全表达式求值</li>
                <li><strong style={{ color:'var(--color-mixed)' }}>复合问题</strong> — 分解分派综合</li>
              </ul>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
};
export default ChatPage;
