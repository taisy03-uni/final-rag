// components/Buttons.tsx
"use client";

import { MdMenu } from 'react-icons/md';
import styles from '../chat.module.css';
import buttonStyles from './styles/buttons.module.css';

interface ButtonsProps {
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  currentLanguage: string;
  setCurrentLanguage: (language: string) => void;
}

const Buttons: React.FC<ButtonsProps> = ({
  isSidebarOpen,
  toggleSidebar,
  currentLanguage,
  setCurrentLanguage,
}) => {
  return (
    <>
      <button className={styles.menuButton} onClick={toggleSidebar} aria-label="Toggle sidebar">
        <MdMenu size={24} />
      </button>
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
    </>
  );
};

export default Buttons;