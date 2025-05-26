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
      messages: [{ text: "Hi there\nHow can I help you today?", isOutgoing: false }]
    }
  ]);
  
  const [activeChatId, setActiveChatId] = useState<string>('1');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const handleNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [{ text: "Hi there\nHow can I help you today?", isOutgoing: false }]
    };
    setChats([...chats, newChat]);
    setActiveChatId(newChat.id);
    setIsSidebarOpen(true); // Open sidebar when creating new chat
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
        <ChatInterface
          currentLanguage="american"
          messages={activeChat.messages}
          onMessagesUpdate={updateChatMessages}
        />
      </div>
    </div>
  );
};

export default Chatbot;