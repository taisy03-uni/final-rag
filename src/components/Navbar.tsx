import Link from 'next/link';
import Image from 'next/image';

export default function Navbar() {
  return (
    <nav className="bg-white p-4 shadow-sm">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-2">
          {/* Logo Image */}
          <div className="h-20 w-20 relative"> {/* make the size bigger*/}
            <Image 
              src="/logo.png" 
              alt="LADA.AI Logo"
              fill
              className="object-contain"
            />
          </div>
          
          {/* Text Logo */}
          <Link href="/" className="text-xl font-bold hover:text-[#354AB8] transition-colors">
            <span className="text-black">LADA.</span>
            <span className="text-[#8396F8]">AI</span>
          </Link>
        </div>

        <div className="space-x-6">
          <Link 
            href="/chat" 
            className="text-black hover:text-[#354AB8] transition-colors"
          >
            Try Out
          </Link>
          <Link 
            href="/about" 
            className="text-black hover:text-[#354AB8] transition-colors"
          >
            About Us
          </Link>
          <Link 
            href="/security" 
            className="text-black hover:text-[#354AB8] transition-colors"
          >
            Security
          </Link>
          <Link 
            href="/login" 
            className="text-black hover:text-[#354AB8] transition-colors"
          >
            Login
          </Link>
        </div>
      </div>
    </nav>
  );
}