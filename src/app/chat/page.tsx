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
  const [currentOutput, setCurrentOutput] = useState<string>('AItext');

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
    setIsLoading(true);
    
    if (currentOutput === 'cases') {
      try {
        const pineconeResponse = await fetch(`/api/pinecone`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: message }),
        });
        if (!pineconeResponse.ok) {
          throw new Error('Failed to query Pinecone');
        }
        interface CaseData {
          _id: string;
          _score: number;
          fields: {
            judgment_date: string;
            name: string;
            text: string;
            uri: string;
          };
        }
        const pineconeData = await pineconeResponse.json();
        const first5Cases: CaseData[] = pineconeData.result.hits.slice(0, 5);

        const caseResponse = `
          <style>
            .case-result {
              border: 1px solid #e0e0e0;
              border-radius: 8px;
              padding: 1rem;
              margin: 1rem 0;
              background: #f9f9f9;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .case-result h3 {
              margin: 0 0 0.5rem 0;
              font-size: 1.1rem;
              color: #333;
            }
            .case-meta {
              color: #666;
              font-size: 0.9rem;
              margin-bottom: 0.5rem;
            }
            .case-link-box {
              margin: 0.75rem 0;
              text-align: center;
            }
            .case-link-box a {
              display: inline-block;
              padding: 0.5rem 1rem;
              background: #354AB8;
              color: white;
              text-decoration: none;
              border-radius: 4px;
              font-weight: 500;
              transition: background 0.2s;
              font-size: 0.9rem;
            }
            .case-link-box a:hover {
              background: #0d4b7a;
            }
            .case-excerpt {
              margin-top: 0.75rem;
              padding: 0.75rem;
              background: #fff;
              border-left: 3px solid #354AB8;
              font-size: 0.85rem;
              color: #555;
              line-height: 1.5;
            }
          </style>

          <div>
            ${first5Cases.map((caseData: CaseData, index: number) => {
              const cleanUri = caseData.fields.uri.split('#')[0].replace('/id', '');
              return `
                <div class="case-result">
                  <h3>${caseData.fields.name}</h3>
                  <p class="case-meta"><strong>Judgment Date:</strong> ${caseData.fields.judgment_date}</p>
                  <p class="case-meta"><strong>Score:</strong> ${caseData._score.toFixed(2)}</p>
                  <div class="case-link-box">
                    <a href="${cleanUri}" target="_blank" rel="noopener noreferrer">
                      🔍 View Full Case on The National Archives
                    </a>
                  </div>
                  <div class="case-excerpt">
                    <p><strong>Excerpt:</strong> ${caseData.fields.text}...</p>
                  </div>
                  ${index < first5Cases.length - 1 ? '<div class="case-divider"></div>' : ''}
                </div>
              `;
            }).join('')}
          </div>
        `;

        const botMessages = [{ text: caseResponse, isOutgoing: false }];
        updateChatMessages([...updatedMessages, ...botMessages]);

      } catch (error) {
        updateChatMessages([...updatedMessages, {
          text: "⚠️ Failed to get response. Please try again.",
          isOutgoing: false
        }]);
      }
      finally {
        setIsLoading(false); 
      }
    }
    else if (currentOutput === 'AItext') {
    try {
      // Call both Gemini and OpenAI APIs in parallel
      const [geminiRes, openaiRes] = await Promise.all([
        fetch('/api/gemini', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            messages: updatedMessages,
            currentLanguage
          }),
        }),
        fetch('/api/openai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            messages: updatedMessages,
            currentLanguage
          }),
        })
      ]);
  
      if (!geminiRes.ok || !openaiRes.ok) {
        throw new Error('Failed to get responses from AI services');
      }
      
      const geminiData = await geminiRes.json();
      const openaiData = await openaiRes.json();
      
      // Combine both responses
      const combinedResponse = `
## Gemini Analysis
${geminiData.response}

---

## OpenAI Search & Analysis
${openaiData.response}
      `;
      
      const botMessages = [{ text: combinedResponse, isOutgoing: false }];
      updateChatMessages([...updatedMessages, ...botMessages]);
  
    } catch (error) {
      updateChatMessages([...updatedMessages, {
        text: "⚠️ Failed to get response. Please try again.",
        isOutgoing: false
      }]);
    } finally {
      setIsLoading(false); 
    }
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
          <Buttons isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar} currentLanguage={currentLanguage} setCurrentLanguage={setCurrentLanguage} currentOutput= {currentOutput} setCurrentOutput={setCurrentOutput} />
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