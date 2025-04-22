import React from 'react';
import styles from './chat.module.css';
import { MdSmartToy, MdSend } from 'react-icons/md';
import ChatInterface from './components/ChatInterface';
import Buttons from './components/Buttons';

const Chatbot: React.FC = () => {
  return (
    <div className={styles.chatbot}>
      <ul className={styles.chatbox}>
        <li className={`${styles.chat} ${styles.incoming}`}>
          <span className={styles.icon}><MdSmartToy /></span>
          <p>Hi there <br />How can I help you today?</p>
        </li>
      </ul>
      <div className={styles.inputContainer}>
        <div className={styles.inputWrapper}>
          <textarea 
            placeholder="Enter a message..." 
            spellCheck="false" 
            required
            className={styles.textarea}
          ></textarea>
          <button className={styles.sendBtn}>
            <MdSend />
          </button>
        </div>
      </div>
      <div className={styles.languageSelector}>
        <span>Response Language:</span>
        <button className={`${styles.langBtn} ${styles.active}`} data-lang="american">American English</button>
        <button className={styles.langBtn} data-lang="british">British English</button>
      </div>
    </div>
  );
};

export default Chatbot;