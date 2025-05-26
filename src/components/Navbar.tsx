import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="bg-transparent backdrop-blur-sm p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link 
          href="/" 
          className="text-3xl font-extrabold tracking-tight relative group px-4 py-2"
        >
          <span className="relative z-10 group-hover:text-white transition-colors duration-300">
            <span>LADA</span>
            <span>.AI</span>
          </span>
          <div className="absolute inset-0 bg-black scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
        </Link>

        <div className="flex items-center space-x-4">
          <Link 
            href="/chat" 
            className="relative bg-black text-white px-6 py-2 rounded-lg font-medium overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(255,255,255,0.6)]"
          >
            <span className="relative z-10">Try Out</span>
          </Link>
          <Link 
            href="/pricing" 
            className="relative px-4 py-2 text-black font-medium rounded-lg overflow-hidden transition-all duration-300 hover:scale-105 hover:bg-black hover:text-white hover:shadow-[0_0_20px_rgba(255,255,255,0.6)]"
          >
            <span className="relative z-10">Pricing</span>
          </Link>
          <Link 
            href="/security" 
            className="relative px-4 py-2 text-black font-medium rounded-lg overflow-hidden transition-all duration-300 hover:scale-105 hover:bg-black hover:text-white hover:shadow-[0_0_20px_rgba(255,255,255,0.6)]"
          >
            <span className="relative z-10">Security</span>
          </Link>
          <Link 
            href="/login" 
            className="relative border-2 border-black text-black px-6 py-2 rounded-lg font-medium overflow-hidden transition-all duration-300 hover:scale-105 hover:bg-black hover:text-white hover:shadow-[0_0_20px_rgba(255,255,255,0.6)]"
          >
            <span className="relative z-10">Login</span>
          </Link>
        </div>
      </div>
    </nav>
  );
}