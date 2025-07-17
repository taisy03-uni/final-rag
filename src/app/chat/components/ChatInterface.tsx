"use client";

import React, { useState, useRef, useEffect } from 'react';
import { MdSmartToy, MdSend, MdUpload, MdHelp, MdQuestionAnswer } from 'react-icons/md';
import styles from '../chat.module.css';
import loadingStyles from './styles/loadingDots.module.css';
import newChatStyles from './styles/newChat.module.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import squeeze from 'remark-squeeze-paragraphs';

interface ChatInterfaceProps {
  currentLanguage: string;
  messages: Array<{ text: string; isOutgoing: boolean }>;
  onMessagesUpdate: (messages: Array<{ text: string; isOutgoing: boolean }>) => void;
  onSendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
  onFileUpload?: (file: File) => void; // Add this prop if you need file upload functionality
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  currentLanguage, 
  messages, 
  onMessagesUpdate, 
  onSendMessage,
  isLoading,
  onFileUpload
}) => {
  const [userMessage, setUserMessage] = useState<string>('');
  const chatboxRef = useRef<HTMLUListElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isNewChat = messages.length === 0;

  const handleChat = async () => {
    const message = userMessage.trim();
    if (!message) return;
    
    const newMessages = [...messages, { text: message, isOutgoing: true }];
    onMessagesUpdate(newMessages);
    setUserMessage('');

    try {
      await onSendMessage(message);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessages = [...newMessages, { 
        text: "Sorry, I couldn't process your request. Please try again.", 
        isOutgoing: false 
      }];
      onMessagesUpdate(errorMessages);
    }
  };

  const handleQuickAction = (action: string) => {
    let message = '';
    switch (action) {
      case 'learnMore':
        message = "Hi xxxx, We are a legal research assistant specialized in UK law, providing insights and guidance based on the latest legal context. We assist in researching case files, answering general legal questions, and offering professional advice. How can I assist you today?";
        onMessagesUpdate([...messages, { text: message, isOutgoing: false }]);
        handleChat(); 
        break;
      case 'askQuestion':
        message = "I have a general question.";
        setUserMessage(message);
        break;
      case 'caseResearch':
        message = "I want to upload a case file for research.";
        if (fileInputRef.current) {
          fileInputRef.current.click(); // Trigger file upload dialog
        }
        break;
      default:
        return;
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (onFileUpload) {
        onFileUpload(file);
      }
      handleQuickAction('caseResearch');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
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
      chatboxRef.current.scrollTo({
        top: chatboxRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages]);

  return (
    <div className={styles.chatbotInterface}>
      {isNewChat && !isLoading ? (
        <div className={newChatStyles.welcomeScreen}>
          <h2>How can I help you today?</h2>
          <div className={newChatStyles.quickActions}>
            <button className={newChatStyles.quickActionBtn} onClick={() => handleQuickAction('learnMore')}>
            <MdHelp size={24} />
            <span>Learn more about this tool</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf,.doc,.docx,.txt"
              style={{ display: 'none' }}
            />
            <button className={newChatStyles.quickActionBtn}onClick={() => fileInputRef.current?.click()}>
            <MdUpload size={24} />
            <span>Upload a case file</span>
            </button>
            <button className={newChatStyles.quickActionBtn} onClick={() => handleQuickAction('askQuestion')} >
            <MdQuestionAnswer size={24} />
            <span>Ask a general question</span>
            </button>
          </div>
        </div>
      ) : (
        <ul className={styles.chatbox} ref={chatboxRef}>
          {messages.map((msg, index) => (
            <li key={index} className={`${styles.chat} ${msg.isOutgoing ? styles.outgoing : styles.incoming}`}>
              {!msg.isOutgoing && <span className={styles.icon}><MdSmartToy /></span>}
              <div className={styles.chatContent}> 
              {msg.text.includes('<div class="case-result">') ? (
                <div 
                  dangerouslySetInnerHTML={{ __html: msg.text }}
                  className={styles.caseResult}
                />
              ) : (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, squeeze]}
                  components={{
                    h1: ({node, ...props}) => <h1 style={{ fontSize: '2em', fontWeight: 'bold' }} {...props} />,
                    h2: ({node, ...props}) => <h2 style={{ fontSize: '1.5em', fontWeight: 'bold' }} {...props} />,
                    h3: ({node, ...props}) => <h3 style={{ fontSize: '1.2em', fontWeight: 'bold' }} {...props} />,
                    h4: ({node, ...props}) => <h4 style={{ fontSize: '1em', fontWeight: 'bold' }} {...props} />
                  }}>
                  {msg.text}
                </ReactMarkdown>
                )}
              </div>
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
      )}
      
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