"use client";

import React, { useState, useRef, useEffect } from 'react';
import { MdSmartToy, MdSend } from 'react-icons/md';
import styles from '../chat.module.css';

interface ChatInterfaceProps {
  currentLanguage: string;
  messages: Array<{ text: string; isOutgoing: boolean }>;
  onMessagesUpdate: (messages: Array<{ text: string; isOutgoing: boolean }>) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ currentLanguage, messages, onMessagesUpdate }) => {
  const [userMessage, setUserMessage] = useState<string>('');
  const chatboxRef = useRef<HTMLUListElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChat = () => {
    console.log('handleChat called', userMessage); // Debug log
    const message = userMessage.trim();
    if (!message) {
      console.log('Empty message, returning'); // Debug log
      return;
    }
    
    // Add user message to chat
    const newMessages = [...messages, { text: message, isOutgoing: true }];
    onMessagesUpdate(newMessages);
    
    setUserMessage(''); // Clear input
    
    // Add bot's "thinking" message
    setTimeout(() => {
      const withThinking = [...newMessages, { text: "Let me think about that...", isOutgoing: false }];
      onMessagesUpdate(withThinking);
      
      // Simulate bot response after 1.5 seconds
      setTimeout(() => {
        // Remove thinking message and add actual response
        const finalMessages = [...newMessages, { 
          text: "I'm a demo bot for now. Soon I'll be connected to the Gemini API to provide real responses!", 
          isOutgoing: false 
        }];
        onMessagesUpdate(finalMessages);
      }, 1500);
    }, 500);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && window.innerWidth > 800) {
      e.preventDefault();
      handleChat();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [userMessage]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    console.log('Messages updated:', messages); // Debug log
    if (chatboxRef.current) {
      chatboxRef.current.scrollTo(0, chatboxRef.current.scrollHeight);
    }
  }, [messages]);

  return (
    <div className={styles.chatbot}>
      <ul className={styles.chatbox} ref={chatboxRef}>
        {messages.map((msg, index) => (
          <li key={index} className={`${styles.chat} ${msg.isOutgoing ? '' : styles.incoming}`}>
            {!msg.isOutgoing && <span className={styles.icon}><MdSmartToy /></span>}
            <p>{msg.text}</p>
          </li>
        ))}
      </ul>
      <div className={styles.inputContainer}>
        <div className={styles.inputWrapper}>
          <textarea 
            ref={textareaRef}
            placeholder="Enter a message..." 
            spellCheck="false" 
            required
            className={styles.textarea}
            value={userMessage}
            onChange={(e) => {
              console.log('Input changed:', e.target.value); // Debug log
              setUserMessage(e.target.value);
            }}
            onKeyDown={handleKeyDown}
          ></textarea>
          <button 
            className={styles.sendBtn} 
            onClick={() => {
              console.log('Send button clicked'); // Debug log
              handleChat();
            }}
          >
            <MdSend size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;