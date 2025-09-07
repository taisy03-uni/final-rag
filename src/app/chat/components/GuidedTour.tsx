import React, { useState, useEffect } from 'react';
import styles from './styles/buttons.module.css';

interface GuidedTourStep {
  buttonId: string; // unique ID for each button
  text: string; // text to display
}

interface GuidedTourProps {
  steps: GuidedTourStep[];
  restartSignal?: number; 
  stepDuration?: number; // duration for each step in ms
}

const GuidedTour: React.FC<GuidedTourProps> = ({ steps, restartSignal, stepDuration = 3000 }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [showText, setShowText] = useState(false);
  const [textPosition, setTextPosition] = useState<{ top: number; left: number }>({ top: 0, left: 0 });

  useEffect(() => {
    setCurrentStep(0); // restart tour
    }, [restartSignal]);

  useEffect(() => {
    if (currentStep >= steps.length) return;
  
    setShowText(false);

    const step = steps[currentStep];

    // If step is "justText", center the text in the screen
    if (step.buttonId === "justText") {
      setTextPosition({
        top: window.innerHeight / 4,
        left: window.innerWidth / 2,

      });
      setTimeout(() => setShowText(true), 500);

      const stepTimer = setTimeout(() => {
        setShowText(false);
        setCurrentStep(prev => prev + 1);
      }, stepDuration);

      return () => clearTimeout(stepTimer);
    };
  
    const button = document.getElementById(step.buttonId);
    if (button)  {
      // Highlight button
      button.style.zIndex = '10000';
      button.style.position = 'relative';
      button.style.borderRadius = '8px';
      button.style.boxShadow = `
        0 0 8px #354AB8,
        0 0 16px #354AB8
      `;
  
      // Calculate position for text relative to button
      const rect = button.getBoundingClientRect();
      const tooltipHeight = 80; // estimated tooltip height (px), adjust if needed
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
  
      let topPosition: number;
      if (spaceBelow > tooltipHeight) {
        // enough space below → place tooltip below
        topPosition = rect.bottom + window.scrollY + 8;
      } else {
        // not enough space below → place tooltip above
        topPosition = rect.top + window.scrollY - tooltipHeight - 8;
      }
  
      setTextPosition({
        top: topPosition,
        left: Math.min(
          Math.max(rect.left + window.scrollX + rect.width / 2, 150),
          window.innerWidth - 150
        ),
      });
    }
  
    const textTimer = setTimeout(() => setShowText(true), 500);
  
    const stepTimer = setTimeout(() => {
      if (button) {
        button.style.boxShadow = '';
        button.style.zIndex = '';
        button.style.position = '';
      }
      setShowText(false);
      setCurrentStep(prev => prev + 1);
    }, stepDuration);
  
    return () => {
      clearTimeout(textTimer);
      clearTimeout(stepTimer);
      if (button) {
        button.style.boxShadow = '';
        button.style.zIndex = '';
        button.style.position = '';
      }
    };
  }, [currentStep, stepDuration, steps]);
  

  if (currentStep >= steps.length) return null;
  
  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'transparent',
        pointerEvents: 'none',
        zIndex: 9998,
      }}
    >
      {showText && (
        <div
          style={{
            position: 'absolute',
            top: textPosition.top,
            left: textPosition.left,
            transform: 'translateX(-50%)', // center text horizontally under button
            maxWidth: '300px',
            backgroundColor: 'rgba(0,0,0,0.8)',
            padding: '0.8rem 1rem',
            borderRadius: '6px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            fontSize: '1rem',
            lineHeight: 1.4,
            textAlign: 'center',
            color: '#fff',
            zIndex: 10001,
          }}
        >
          {steps[currentStep].text}
        </div>
      )}
    </div>
  );
};

export default GuidedTour;
