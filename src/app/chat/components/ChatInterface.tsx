import React, { useState, useRef, useEffect } from 'react';
import { MdSmartToy } from 'react-icons/md';
import styles from '../chat.module.css';

interface ChatInterfaceProps {
  currentLanguage: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ currentLanguage }) => {
  const [userMessage, setUserMessage] = useState<string>('');
  const [messages, setMessages] = useState<Array<{
    text: string;
    isOutgoing: boolean;
  }>>([
    { text: "Hi there\nHow can I help you today?", isOutgoing: false }
  ]);
  const chatboxRef = useRef<HTMLUListElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const API_KEY = "AIzaSyBwv5r749w6WcU6cfkPp7GdliKpPG3hpAg";
  const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${API_KEY}`;

  const getLanguageInstruction = () => {
    return currentLanguage === 'british' 
      ? " [Please respond in British English using British spelling and terminology.]"
      : "";
  };

  const generateResponse = async (message: string) => {
    const languageInstruction = getLanguageInstruction();
    const fullPrompt = message + languageInstruction;
    
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        contents: [{ 
          role: "user", 
          parts: [{ text: fullPrompt }] 
        }] 
      }),
    };
    
    try {
      const response = await fetch(API_URL, requestOptions);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error?.message || 'Unknown error');
      
      const responseText = data.candidates[0].content.parts[0].text.replace(/\*\*(.*?)\*\*/g, '$1');
      setMessages(prev => [...prev, { text: responseText, isOutgoing: false }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        text: error instanceof Error ? error.message : 'An error occurred', 
        isOutgoing: false 
      }]);
    }
  };

  const handleChat = () => {
    const message = userMessage.trim();
    if (!message) return;
    
    setUserMessage('');
    setMessages(prev => [...prev, { text: message, isOutgoing: true }]);
    
    setTimeout(() => {
      setMessages(prev => [...prev, { text: "Thinking...", isOutgoing: false }]);
      generateResponse(message);
    }, 600);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && window.innerWidth > 800) {
      e.preventDefault();
      handleChat();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [userMessage]);

  useEffect(() => {
    if (chatboxRef.current) {
      chatboxRef.current.scrollTo(0, chatboxRef.current.scrollHeight);
    }
  }, [messages]);

  return (
    <>
      <ul className={styles.chatbox} ref={chatboxRef}>
        {messages.map((msg, index) => (
          <li key={index} className={`${styles.chat} ${msg.isOutgoing ? '' : styles.incoming}`}>
            {!msg.isOutgoing && <span className={styles.icon}><MdSmartToy /></span>}
            <p>{msg.text}</p>
          </li>
        ))}
      </ul>
      <div className={styles.chatbot}>
        <div className={styles.chatbot}>
          <textarea 
            ref={textareaRef}
            placeholder="Enter a message..." 
            spellCheck="false" 
            required
            className={styles.textarea}
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyDown={handleKeyDown}
          ></textarea>
          <button className={styles.sendBtn} onClick={handleChat}>
            {/* MdSend will be imported in page.tsx */}
          </button>
        </div>
      </div>
    </>
  );
};

export default ChatInterface;