"use client";

import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import { MdMenu } from 'react-icons/md';
import styles from './chat.module.css';

const Chatbot: React.FC = () => {
  const [chats, setChats] = useState<Array<{
    id: string;
    title: string;
    messages: Array<{ text: string; isOutgoing: boolean }>;
  }>>([
    {
      id: '1',
      title: 'New Chat',
      messages: [] // Start with empty messages
    }
  ]);
  
  const [activeChatId, setActiveChatId] = useState<string>('1');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentLanguage, setCurrentLanguage] = useState<string>('american');
  const [isLoading, setIsLoading] = useState(false);

  const handleNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [] // Empty for new chats
    };
    setChats([...chats, newChat]);
    setActiveChatId(newChat.id);
    setIsSidebarOpen(true);
  };

  const activeChat = chats.find(chat => chat.id === activeChatId) || chats[0];

  const updateChatMessages = (messages: Array<{ text: string; isOutgoing: boolean }>) => {
    setChats(prevChats => 
      prevChats.map(chat => 
        chat.id === activeChatId 
          ? { ...chat, messages, title: messages.length > 1 ? messages[1].text.slice(0, 30) + '...' : chat.title }
          : chat
      )
    );
  };

  const handleSendMessage = async (message: string) => {
    const userMessage = { text: message, isOutgoing: true };
    const updatedMessages = [...activeChat.messages, userMessage];
    updateChatMessages(updatedMessages);
    
    setIsLoading(true);
  
    try {
      const res = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: updatedMessages,
          isFirstMessage: activeChat.messages.length === 0 // Flag for first message
        }),
      });
  
      if (!res.ok) throw new Error(res.statusText);
  
      const { response } = await res.json();
      
      // Add welcome message only if it's the first exchange
      const botMessages = activeChat.messages.length === 0
        ? [
            { text: response, isOutgoing: false }
          ]
        : [{ text: response, isOutgoing: false }];
      
      updateChatMessages([...updatedMessages, ...botMessages]);
  
    } catch (error) {
      updateChatMessages([...updatedMessages, {
        text: "⚠️ Failed to get response. Please try again.",
        isOutgoing: false
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className={styles.container}>
      <div className={`${styles.sidebarContainer} ${isSidebarOpen ? styles.open : ''}`}>
        <Sidebar
          chats={chats}
          activeChatId={activeChatId}
          onChatSelect={setActiveChatId}
          onNewChat={handleNewChat}
        />
      </div>
      <button className={styles.menuButton} onClick={toggleSidebar} aria-label="Toggle sidebar">
        <MdMenu size={24} />
      </button>
      <div className={styles.chatContainer}>
        <div className={styles.languageSelector}>
          <button 
            className={`${styles.langBtn} ${currentLanguage === 'american' ? styles.active : ''}`}
            onClick={() => setCurrentLanguage('american')}
          >
            American
          </button>
          <button 
            className={`${styles.langBtn} ${currentLanguage === 'british' ? styles.active : ''}`}
            onClick={() => setCurrentLanguage('british')}
          >
            British
          </button>
        </div>
        <ChatInterface
          currentLanguage={currentLanguage}
          messages={activeChat.messages}
          onMessagesUpdate={updateChatMessages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
};

export default Chatbot;