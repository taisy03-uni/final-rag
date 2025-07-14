"use client";

import React, { useState, useRef, useEffect } from 'react';
import { MdSmartToy, MdSend } from 'react-icons/md';
import styles from '../chat.module.css';
import loadingStyles from './styles/loadingDots.module.css';

interface ChatInterfaceProps {
  currentLanguage: string;
  messages: Array<{ text: string; isOutgoing: boolean }>;
  onMessagesUpdate: (messages: Array<{ text: string; isOutgoing: boolean }>) => void;
  onSendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  currentLanguage, 
  messages, 
  onMessagesUpdate, 
  onSendMessage,
  isLoading 
}) => {
  const [userMessage, setUserMessage] = useState<string>('');
  const chatboxRef = useRef<HTMLUListElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChat = async () => {
    const message = userMessage.trim();
    if (!message) return;
    
    // Add user message to chat immediately
    const newMessages = [...messages, { text: message, isOutgoing: true }];
    onMessagesUpdate(newMessages);
    setUserMessage('');
    
    // Add loading indicator
    const loadingMessages = [...newMessages, { 
      text: currentLanguage === 'british' 
        ? "Let me think about that..." 
        : "One moment please...", 
      isOutgoing: false 
    }];
    onMessagesUpdate(loadingMessages);

    try {
      // Call the parent component's send message handler which connects to Gemini
      await onSendMessage(message);
      
      // Remove loading message - parent's onMessagesUpdate will handle the actual response
    } catch (error) {
      console.error('Error sending message:', error);
      // Replace loading message with error
      const errorMessages = [...newMessages, { 
        text: "Sorry, I couldn't process your request. Please try again.", 
        isOutgoing: false 
      }];
      onMessagesUpdate(errorMessages);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
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
    if (chatboxRef.current) {
      chatboxRef.current.scrollTo({
        top: chatboxRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages]);

  return (
    <div className={styles.chatbot}>
      <ul className={styles.chatbox} ref={chatboxRef}>
        {messages.map((msg, index) => (
          <li 
            key={index} 
            className={`${styles.chat} ${msg.isOutgoing ? styles.outgoing : styles.incoming}`}
          >
            {!msg.isOutgoing && <span className={styles.icon}><MdSmartToy /></span>}
            <p>{msg.text}</p>
          </li>
        ))}
        {isLoading && (
        <li className={`${styles.chat} ${styles.incoming}`}>
          <span className={styles.icon}><MdSmartToy /></span>
          <p>
            <span className={loadingStyles.dotsContainer}>
              <span className={loadingStyles.dot}></span>
              <span className={loadingStyles.dot}></span>
              <span className={loadingStyles.dot}></span>
            </span>
          </p>
        </li>
      )}
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
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button 
            className={styles.sendBtn} 
            onClick={handleChat}
            disabled={isLoading || !userMessage.trim()}
          >
            <MdSend size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;