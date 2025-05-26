// components/AnimatedBackground.tsx
"use client"; // Required for client-side components in Next.js 13+

import { useEffect } from 'react';

interface AnimatedBackgroundProps {
  className?: string;
}

export default function AnimatedBackground({ className = "h-screen" }: AnimatedBackgroundProps) {
  useEffect(() => {
    // Load the finisher header script dynamically
    const script = document.createElement('script');
    script.src = '/finisher-header.es5.min.js';
    script.async = true;
    script.onload = () => {
      new (window as any).FinisherHeader({
        "count": 5,
        "size": {
          "min": 328,
          "max": 915,
          "pulse": 2
        },
        "speed": {
          "x": {
            "min": 0.6,
            "max": 1.9
          },
          "y": {
            "min": 0.6,
            "max": 2
          }
        },
        "colors": {
          "background": "#75d5ff",
          "particles": [
            "#8396f8",
            "#ffffff",
            "#8396f8"
          ]
        },
        "blending": "lighten",
        "opacity": {
          "center": 0.6,
          "edge": 0
        },
        "skew": 0,
        "shapes": [
          "c"
        ]
      });
    };
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div 
      className={`finisher-header fixed inset-0 -z-10 w-full ${className}`}
      data-testid="animated-background"
    />
  );
}