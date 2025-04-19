// app/page.tsx
import AnimatedBackground from '@/components/AnimatedBackground';

export default function Home() {
  return (
    <div className="relative">
      <AnimatedBackground />
    
      <div className="text-center py-10 relative z-10">
        <h1 className="text-4xl font-bold mb-6">Welcome to LawAI</h1>
        <p className="text-xl text-gray-600">
          Your legal assistant powered by AI
        </p>
      </div>
    </div>
  );
}