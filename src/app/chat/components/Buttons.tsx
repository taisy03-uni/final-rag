import React, { useState } from 'react';
import styles from './chat.module.css'; // Assuming styles are in chat.module.css

interface ButtonsProps {
    onLanguageChange?: (language: 'american' | 'british') => void;
}

const Buttons: React.FC<ButtonsProps> = ({ onLanguageChange }) => {
    const [activeLanguage, setActiveLanguage] = useState<'american' | 'british'>('american');

    const handleLanguageChange = (language: 'american' | 'british') => {
        setActiveLanguage(language);
        onLanguageChange?.(language); // Notify parent component, if a callback is provided.
    };

    return (
        <div className={styles.languageSelector}>
            <span>Response Language:</span>
            <button
                className={`${styles.langBtn} ${activeLanguage === 'american' ? styles.active : ''}`}
                data-lang="american"
                onClick={() => handleLanguageChange('american')}
            >
                American English
            </button>
            <button
                className={`${styles.langBtn} ${activeLanguage === 'british' ? styles.active : ''}`}
                data-lang="british"
                onClick={() => handleLanguageChange('british')}
            >
                British English
            </button>
        </div>
    );
};

export default Buttons;
