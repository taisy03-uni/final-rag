"use client";

import React from 'react';
import { MdAdd, MdChat } from 'react-icons/md';
import styles from '../chat.module.css';

interface SidebarProps {
  onNewChat: () => void;
  chats: Array<{ id: string; title: string }>;
  activeChatId: string | null;
  onChatSelect: (chatId: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  onNewChat,
  chats,
  activeChatId,
  onChatSelect,
}) => {
  return (
    <div className={styles.sidebar}>
      <div className={styles.sidebarHeader}>
        <h2>LADA.AI</h2>
      </div>
      
      <button className={styles.newChatButton} onClick={onNewChat}>
        <MdAdd size={20} />
        New Chat
      </button>
      
      <div className={styles.chatsList}>
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`${styles.chatItem} ${chat.id === activeChatId ? styles.active : ''}`}
            onClick={() => onChatSelect(chat.id)}
          >
            <MdChat size={20} />
            <span>{chat.title}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar; 