import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState({});
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/status`);
      setStatus(res.data);
    } catch {}
  };

  const sendMessage = async () => {
    if (!input) return;
    const userMsg = { role: 'user', content: input };
    setMessages(msgs => [...msgs, userMsg]);
    setInput('');

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: input });
      setMessages(msgs => [...msgs, { role: 'assistant', content: res.data.response }]);
    } catch (e) {
      setMessages(msgs => [...msgs, { role: 'assistant', content: 'Error: ' + e.message }]);
    }
  };

  const uploadPDFs = async (e) => {
    const files = Array.from(e.target.files);
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    
    try {
      await axios.post(`${API_BASE}/upload-pdfs`, formData);
      alert('PDFs uploaded!');
      fetchStatus();
    } catch (e) {
      alert('Upload failed');
    }
  };

  return (
    <div className="app">
      <header>
        <h1>🤖 AI Research Agent</h1>
        <div>Status: {status.ready ? '✅ Ready' : '⏳ Loading'}</div>
      </header>
      
      <div className="chat">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role}>
            <div>{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyPress={e => e.key === 'Enter' && sendMessage()} />
        <button onClick={sendMessage}>Send</button>
        <input type="file" multiple accept=".pdf" onChange={uploadPDFs} />
      </div>
    </div>
  );
}

export default App;
