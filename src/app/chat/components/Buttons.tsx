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
  currentOutput: string;
  setCurrentOutput: (output: string) => void;
}

const Buttons: React.FC<ButtonsProps> = ({
  isSidebarOpen,
  toggleSidebar,
  currentLanguage,
  setCurrentLanguage,
  currentOutput,
 setCurrentOutput
}) => {
  return (
    <>
      <button className={styles.menuButton} onClick={toggleSidebar} aria-label="Toggle sidebar">
        <MdMenu size={24} />
      </button>
      <div className={buttonStyles.languageSelector}>
        <button 
          className={`${buttonStyles.langBtn} ${currentLanguage === 'american' ? buttonStyles.active : ''}`}
          onClick={() => setCurrentLanguage('american')}
        >
          American
        </button>
        <button 
          className={`${buttonStyles.langBtn} ${currentLanguage === 'british' ? buttonStyles.active : ''}`}
          onClick={() => setCurrentLanguage('british')}
        >
          British
        </button>
        <div className={buttonStyles.outputToggleContainer}>
          <label className={buttonStyles.toggleSwitch}>
            <input 
              type="checkbox"
              checked={currentOutput === 'cases'}
              onChange={() => setCurrentOutput(currentOutput === 'AItext' ? 'cases' : 'AItext')}
            />
            <span className={buttonStyles.slider}>
              <span className={`${buttonStyles.toggleOption} ${currentOutput === 'AItext' ? buttonStyles.activeOption : ''}`}>
                Text
              </span>
              <span className={`${buttonStyles.toggleOption} ${currentOutput === 'cases' ? buttonStyles.activeOption : ''}`}>
                Markdown
              </span>
            </span>
          </label>
        </div>
      </div>
    </>
  );
};

export default Buttons;