"use client";

import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import Buttons from './components/Buttons'; 
import Sidebar from './components/Sidebar';
import { MdMenu } from 'react-icons/md';
import styles from './chat.module.css';

const Chatbot: React.FC = () => {
  const [chats, setChats] = useState<Array<{
    id: string;
    title: string;
    messages: Array<{ text: string; isOutgoing: boolean }>;
    isLoading: boolean;
  }>>([
    {
      id: '1',
      title: 'New Chat',
      messages: [], // Start with empty messages
      isLoading: false
    }
  ]);
  
  const [activeChatId, setActiveChatId] = useState<string>('1');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentLanguage, setCurrentLanguage] = useState<string>('american');

  const handleNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [], // Empty for new chats
      isLoading: false
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
  const setIsLoading = (loading: boolean) => {
    setChats(prevChats => 
      prevChats.map(chat => 
        chat.id === activeChatId ? { ...chat, isLoading: loading } : chat
      )
    );
  }

  const handleSendMessage = async (message: string) => {
    const userMessage = { text: message, isOutgoing: true };
    const updatedMessages = [...activeChat.messages, userMessage];
    updateChatMessages(updatedMessages);
    setIsLoading(true); // Set loading state for the active chat
  
    try {
      const res = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: updatedMessages
        }),
      });
  
      if (!res.ok) throw new Error(res.statusText);
      const { response } = await res.json();
      const botMessages = [{ text: response, isOutgoing: false }];
      
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
          <Sidebar chats={chats} activeChatId={activeChatId} onChatSelect={setActiveChatId} onNewChat={handleNewChat}/>
        </div>
          <Buttons isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar} currentLanguage={currentLanguage} setCurrentLanguage={setCurrentLanguage} />
        <div className={styles.chatContainer}>
          <ChatInterface
            currentLanguage={currentLanguage}
            messages={activeChat.messages}
            onMessagesUpdate={updateChatMessages}
            onSendMessage={handleSendMessage}
            isLoading={activeChat.isLoading} 
          />
        </div>
      </div>
  );
};

export default Chatbot;