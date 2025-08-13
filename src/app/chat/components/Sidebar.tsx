"use client";

import React from 'react';
import { MdAdd, MdChat } from 'react-icons/md';
import sidebarStyles from './styles/sidebar.module.css';

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
  const logoRef = React.createRef<HTMLHeadingElement>();
  const [isHovered, setIsHovered] = React.useState(false);

  const handleLogoClick = () => {
    window.location.href = '/';
  };

  React.useEffect(() => {
    const logoElement = logoRef.current;
    if (logoElement) {
      logoElement.addEventListener('click', handleLogoClick);
      logoElement.addEventListener('mouseenter', () => setIsHovered(true));
      logoElement.addEventListener('mouseleave', () => setIsHovered(false));
      
      return () => {
        logoElement.removeEventListener('click', handleLogoClick);
        logoElement.removeEventListener('mouseenter', () => setIsHovered(true));
        logoElement.removeEventListener('mouseleave', () => setIsHovered(false));
      };
    }
  }, [logoRef]);

  return (
    <div className={sidebarStyles.sidebar}>
      <div className={sidebarStyles.sidebarHeader}>
        <h2 
          ref={logoRef} 
          className={`${sidebarStyles.logoLink} ${isHovered ? sidebarStyles.logoHover : ''}`}
        >
          L.RAG
        </h2>
      </div>
      
      <button className={sidebarStyles.newChatButton} onClick={onNewChat}>
        <MdAdd size={20} />
        New Chat
      </button>
      
      <div className={sidebarStyles.chatsList}>
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`${sidebarStyles.chatItem} ${chat.id === activeChatId ? sidebarStyles.active : ''}`} onClick={() => onChatSelect(chat.id)}>
            <MdChat size={20} />
            <span>{chat.title}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;