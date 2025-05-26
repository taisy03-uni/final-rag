"use client";

import React from 'react';
import { useRouter } from 'next/navigation';
import { MdClose, MdArrowForward } from 'react-icons/md';
import styles from '../chat.module.css';

interface UpgradeModalProps {
  onClose: () => void;
}

const UpgradeModal: React.FC<UpgradeModalProps> = ({ onClose }) => {
  const router = useRouter();

  const handleUpgrade = () => {
    router.push('/pricing');
  };

  return (
    <div className={styles.modalOverlay}>
      <div className={styles.modal}>
        <button className={styles.closeButton} onClick={onClose}>
          <MdClose size={24} />
        </button>
        
        <h2>Upgrade to Continue</h2>
        <p>You've reached the limit of 3 free queries. Upgrade now to unlock:</p>
        
        <ul className={styles.benefitsList}>
          <li>✓ Unlimited AI Queries</li>
          <li>✓ Multiple Chat Sessions</li>
          <li>✓ Priority Response Time</li>
          <li>✓ Advanced Features</li>
        </ul>

        <div className={styles.modalButtons}>
          <button className={styles.upgradeButton} onClick={handleUpgrade}>
            View Pricing Plans <MdArrowForward />
          </button>
        </div>
      </div>
    </div>
  );
};

export default UpgradeModal; 