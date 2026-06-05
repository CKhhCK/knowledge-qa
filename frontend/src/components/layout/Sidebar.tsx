/**
 * Sidebar — 会话列表 + 知识库管理
 */
import React, { useState, useEffect, useRef } from 'react';
import { useChatStore } from '../../store/chatStore';
import { listSessions, clearSession } from '../../services/api';

interface DocInfo {
  document_id: string; title: string; chunk_count: number;
  ingestion_time_ms: number; char_count?: number;
}

const Sidebar: React.FC = () => {
  const [tab, setTab] = useState<'chats'|'knowledge'>('chats');
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [chunkView, setChunkView] = useState<{title:string,chunks:string[],strategy?:string}|null>(null);
  const [chunkStrategy, setChunkStrategy] = useState<string>('recursive');
  const fileRef = useRef<HTMLInputElement>(null);

  // --- Conversations tab ---
  const conversations = useChatStore(s => s.conversations);
  const currentId = useChatStore(s => s.conversationId);
  const setConversationId = useChatStore(s => s.setConversationId);
  const setConversations = useChatStore(s => s.setConversations);
  const clearMessages = useChatStore(s => s.clearMessages);
  const addConversation = useChatStore(s => s.addConversation);
  const removeConversation = useChatStore(s => s.removeConversation);

  const loadConversations = async () => {
    const res = await fetch('/api/v1/chat/conversations');
    if (res.ok) setConversations((await res.json()).conversations || []);
  };
  useEffect(() => { if (tab === 'chats') loadConversations(); }, [tab]);

  const newChat = async () => {
    const res = await fetch('/api/v1/chat/conversations', { method: 'POST' });
    if (res.ok) {
      const c = await res.json();
      addConversation(c); setConversationId(c.conversation_id); clearMessages();
    }
  };
  const switchChat = async (conv: typeof conversations[0]) => {
    setConversationId(conv.conversation_id); clearMessages();
    const res = await fetch(`/api/v1/chat/${conv.conversation_id}/history`);
    if (res.ok) {
      const d = await res.json();
      for (const m of d.messages || []) {
        if (m.role === 'user') useChatStore.getState().addUserMessage(m.content);
        else if (m.role === 'assistant' && m.content) {
          const id = useChatStore.getState().startAssistantMessage();
          useChatStore.getState().appendToAssistantMessage(id, m.content);
          useChatStore.getState().finalizeAssistantMessage(id, m.trace ?? undefined);
        }
      }
    }
  };
  const delChat = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await fetch(`/api/v1/chat/conversations/${id}`, { method: 'DELETE' });
    removeConversation(id);
    if (id === currentId) newChat();
  };

  // --- Knowledge Base tab ---
  const loadDocs = async () => {
    const res = await fetch('/api/v1/documents');
    if (res.ok) setDocs((await res.json()).documents || []);
  };
  useEffect(() => { if (tab === 'knowledge') loadDocs(); }, [tab]);

  const uploadFile = async (file: File) => {
    setUploading(true);
    const fd = new FormData(); fd.append('file', file);
    fd.append('chunk_strategy', chunkStrategy);
    const res = await fetch('/api/v1/documents/upload', { method: 'POST', body: fd });
    setUploading(false);
    if (res.ok) loadDocs();
    else {
      const err = await res.json().catch(() => ({}));
      alert('上传失败: ' + ((err as any).detail || '未知错误'));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  };

  const delDoc = async (id: string) => {
    await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' });
    setDocs(d => d.filter(x => x.document_id !== id));
  };

  // ================================================================

  return (
    <div style={{
      width: '280px', height: '100%', background: 'var(--color-bg-secondary)',
      borderRight: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column',
    }}>
      {/* Tab bar */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--color-border)' }}>
        {[
          { key: 'chats', label: '会话' },
          { key: 'knowledge', label: '知识库' },
        ].map(t => (
          <button key={t.key} onClick={() => setTab(t.key as any)}
            style={{
              flex: 1, padding: '14px 0', border: 'none', cursor: 'pointer',
              background: tab === t.key ? 'var(--color-bg)' : 'transparent',
              color: tab === t.key ? 'var(--color-text)' : 'var(--color-text-muted)',
              fontWeight: tab === t.key ? 600 : 400, fontSize: '0.9em',
              borderBottom: tab === t.key ? '2px solid var(--color-primary)' : '2px solid transparent',
              transition: 'all 0.2s',
            }}
          >{t.label}</button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'chats' ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '12px 16px' }}>
            <button onClick={newChat}
              style={{ width:'100%', padding:'10px', borderRadius:'var(--radius-md)',
                border:'1px solid var(--color-border)', background:'var(--color-bg)',
                color:'var(--color-text)', cursor:'pointer', fontSize:'0.9em',
                fontWeight:500, transition:'all 0.2s' }}
            >+ 新会话</button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px' }}>
            {conversations.map(c => (
              <div key={c.conversation_id} onClick={() => switchChat(c)}
                style={{ padding:'12px', marginBottom:'2px', borderRadius:'var(--radius-sm)',
                  cursor:'pointer', transition:'all 0.15s',
                  background: c.conversation_id===currentId?'var(--color-bg-tertiary)':'transparent',
                  border: c.conversation_id===currentId?'1px solid var(--color-border)':'1px solid transparent',
                }}>
                <div style={{ display:'flex',justifyContent:'space-between',alignItems:'center' }}>
                  <div style={{ flex:1,minWidth:0 }}>
                    <div style={{ fontSize:'0.85em',fontWeight:500,color:'var(--color-text)',
                      overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap' }}>
                      {c.title||'新对话'}</div>
                    <div style={{ fontSize:'0.75em',color:'var(--color-text-muted)',marginTop:'2px' }}>
                      {c.message_count} 条消息</div>
                  </div>
                  <span onClick={e=>delChat(c.conversation_id,e)}
                    style={{ color:'var(--color-text-muted)',cursor:'pointer',fontSize:'0.8em',
                      opacity:0.5,padding:'2px 6px' }}>✕</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Knowledge Base tab */
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Strategy selector */}
          <div style={{ padding: '12px 12px 0' }}>
            <label style={{ fontSize:'0.8em',color:'var(--color-text-secondary)',display:'block',marginBottom:'4px' }}>分块策略</label>
            <select value={chunkStrategy} onChange={e => setChunkStrategy(e.target.value)}
              style={{ width:'100%',padding:'8px 10px',borderRadius:'var(--radius-sm)',
                border:'1px solid var(--color-border)',background:'var(--color-bg)',
                color:'var(--color-text)',fontSize:'0.85em',cursor:'pointer',outline:'none' }}>
              <option value="recursive">递归语义（推荐）</option>
              <option value="markdown">Markdown 标题</option>
              <option value="fixed">固定字符</option>
            </select>
          </div>

          {/* Upload area */}
          <div
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              margin: '12px', padding: '24px 16px',
              border: `2px dashed ${dragOver ? 'var(--color-primary)' : 'var(--color-border)'}`,
              borderRadius: 'var(--radius-md)', textAlign: 'center', cursor: 'pointer',
              background: dragOver ? 'rgba(99,102,241,0.05)' : 'var(--color-bg)',
              transition: 'all 0.2s',
            }}
          >
            <div style={{ fontSize: '1.8em', marginBottom: '8px' }}>
              {uploading ? '⏳' : '📁'}
            </div>
            <div style={{ fontSize: '0.85em', color: 'var(--color-text-secondary)', fontWeight: 500 }}>
              {uploading ? '上传中...' : '点击或拖拽文件上传'}
            </div>
            <div style={{ fontSize: '0.75em', color: 'var(--color-text-muted)', marginTop: '4px' }}>
              支持 PDF, TXT, MD, DOCX, CSV, JSON
            </div>
            <input ref={fileRef} type="file" hidden onChange={e => {
              const f = e.target.files?.[0]; if (f) uploadFile(f); e.target.value = '';
            }} />
          </div>

          {/* Document list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 12px' }}>
            {docs.length === 0 && !uploading && (
              <div style={{ textAlign: 'center', padding: '20px', color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                暂无文档，上传一个试试
              </div>
            )}
            {docs.map(d => (
              <div key={d.document_id} style={{
                padding: '10px 12px', marginBottom: '4px', borderRadius: 'var(--radius-sm)',
                background: 'var(--color-bg)', border: '1px solid var(--color-border)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '0.85em', fontWeight: 500, color: 'var(--color-text)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    📄 {d.title || d.document_id}</div>
                  <div style={{ fontSize: '0.75em', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                    {d.chunk_count || '-'} 分块 · {d.chunk_strategy || 'recursive'}
                  </div>
                </div>
                <div style={{ display:'flex',gap:'4px',flexShrink:0,marginLeft:'8px' }}>
                  <button onClick={async () => {
                    const res = await fetch(`/api/v1/documents/${d.document_id}/chunks`);
                    if (res.ok) {
                      const data = await res.json();
                      setChunkView({ title: d.title||d.document_id, chunks: data.chunks||[],
                                     strategy: d.chunk_strategy || 'recursive' });
                    }
                  }}
                    style={{ background:'none',border:'none',color:'var(--color-primary)',
                      cursor:'pointer',fontSize:'0.75em',padding:'2px 6px',opacity:0.8 }}>
                    查看
                  </button>
                  <button onClick={() => delDoc(d.document_id)}
                    style={{ background:'none',border:'none',color:'var(--color-error)',
                      cursor:'pointer',fontSize:'0.75em',padding:'2px 6px',opacity:0.7 }}>
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chunk viewer modal */}
      {chunkView && (
        <div onClick={() => setChunkView(null)} style={{
          position:'fixed',top:0,left:0,right:0,bottom:0,background:'rgba(0,0,0,0.7)',
          zIndex:1000,display:'flex',alignItems:'center',justifyContent:'center',
        }}>
          <div onClick={e => e.stopPropagation()} style={{
            background:'var(--color-bg-secondary)',border:'1px solid var(--color-border)',
            borderRadius:'var(--radius-lg)',width:'700px',maxHeight:'80vh',
            display:'flex',flexDirection:'column',
          }}>
            <div style={{ padding:'16px 20px',borderBottom:'1px solid var(--color-border)',
              display:'flex',justifyContent:'space-between',alignItems:'center' }}>
              <div style={{ fontWeight:600,fontSize:'1em',color:'var(--color-text)' }}>
                📄 {chunkView.title} — {chunkView.chunks.length} 个分块
                <span style={{ fontSize:'0.8em',color:'var(--color-text-muted)',fontWeight:400,marginLeft:'8px' }}>
                  ({chunkView.strategy === 'markdown' ? 'Markdown标题' :
                    chunkView.strategy === 'fixed' ? '固定字符' : '递归语义'})
                </span>
              </div>
              <button onClick={() => setChunkView(null)}
                style={{ background:'none',border:'none',color:'var(--color-text-muted)',
                  cursor:'pointer',fontSize:'1.3em' }}>✕</button>
            </div>
            <div style={{ flex:1,overflowY:'auto',padding:'16px 20px' }}>
              {chunkView.chunks.map((chunk, i) => (
                <div key={i} style={{
                  marginBottom:'16px',padding:'14px 16px',
                  background:'var(--color-bg)',borderRadius:'var(--radius-md)',
                  border:'1px solid var(--color-border)',
                }}>
                  <div style={{ fontSize:'0.8em',fontWeight:600,color:'var(--color-primary)',
                    marginBottom:'8px' }}>分块 {i + 1} · {chunk.length} 字符</div>
                  <div style={{ fontSize:'0.85em',color:'var(--color-text-secondary)',
                    lineHeight:1.7,whiteSpace:'pre-wrap',wordBreak:'break-word' }}>
                    {chunk}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
