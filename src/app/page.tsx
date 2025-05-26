// app/page.tsx
import AnimatedBackground from '@/components/AnimatedBackground';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="relative min-h-screen">
      <AnimatedBackground className="h-screen" />
      <div className="relative z-10 container mx-auto px-4 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold mb-4 tracking-tight">
            <span className="text-black">Transform Your Legal Work with </span>
            <span className="text-black">AI</span>
          </h1>
          
          <p className="text-2xl text-black mb-8 leading-relaxed">
            Streamline research, automate documents, and enhance your legal practice
          </p>
          
          <p className="text-xl text-black mb-6 leading-relaxed font-medium">
            LADA.AI is your advanced legal assistant, designed specifically for lawyers.
            Our platform streamlines multiple tasks including legal research and writing, making your work more efficient than ever.
          </p>
          
          <p className="text-lg text-black mb-12">
            With our unwavering commitment to security, your conversations remain completely anonymous.
            We understand the importance of confidentiality in legal matters and ensure your data stays protected.
          </p>

          <Link 
            href="/chat"
            className="relative inline-block bg-black text-white text-xl font-semibold px-12 py-4 rounded-lg overflow-hidden transition-all duration-300 transform hover:scale-105 hover:shadow-[0_0_20px_rgba(255,255,255,0.6)]"
          >
            <span className="relative z-10">Try Our Bot Now</span>
          </Link>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="bg-white/90 backdrop-blur-sm p-6 rounded-xl shadow-lg">
            <h3 className="text-xl font-semibold mb-3 text-[#354AB8]">Legal Research</h3>
            <p className="text-black">Access comprehensive legal research assistance powered by advanced AI technology.</p>
          </div>
          <div className="bg-white/90 backdrop-blur-sm p-6 rounded-xl shadow-lg">
            <h3 className="text-xl font-semibold mb-3 text-[#354AB8]">Document Writing</h3>
            <p className="text-black">Generate and review legal documents with enhanced efficiency and accuracy.</p>
          </div>
          <div className="bg-white/90 backdrop-blur-sm p-6 rounded-xl shadow-lg">
            <h3 className="text-xl font-semibold mb-3 text-[#354AB8]">Secure Platform</h3>
            <p className="text-black">Your data remains confidential with our enterprise-grade security measures.</p>
          </div>
        </div>
      </div>
    </div>
  );
}