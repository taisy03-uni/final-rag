// app/page.tsx
import AnimatedBackground from '@/components/AnimatedBackground';

export default function Home() {
  return (
    <div className="relative">
      <AnimatedBackground />
    
      <div className="text-center py-10 relative z-10">
      <h1 className="text-8xl font-bold hover:text-[#354AB8] transition-colors">
            <span className="text-black">LADA.</span>
            <span className="text-[#8396F8]">AI</span>
        </h1>
        <p className="text-xl text-gray-600">
        </p>
      </div>
    </div>
  );
}