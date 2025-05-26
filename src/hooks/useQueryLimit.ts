import { useState, useEffect } from 'react';
import { FREE_QUERY_LIMIT } from '@/config/openai';

export const useQueryLimit = () => {
  const [queryCount, setQueryCount] = useState(0);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);

  useEffect(() => {
    // Load query count from localStorage
    const storedCount = localStorage.getItem('queryCount');
    if (storedCount) {
      setQueryCount(parseInt(storedCount, 10));
    }
  }, []);

  const incrementQueryCount = () => {
    const newCount = queryCount + 1;
    setQueryCount(newCount);
    localStorage.setItem('queryCount', newCount.toString());

    if (newCount >= FREE_QUERY_LIMIT) {
      setShowUpgradeModal(true);
      return false; // Indicates that the limit has been reached
    }
    return true; // Indicates that the query can proceed
  };

  const closeUpgradeModal = () => {
    setShowUpgradeModal(false);
  };

  return {
    queryCount,
    showUpgradeModal,
    incrementQueryCount,
    closeUpgradeModal,
    hasReachedLimit: queryCount >= FREE_QUERY_LIMIT
  };
}; 