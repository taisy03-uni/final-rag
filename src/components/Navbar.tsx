import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="bg-gray-800 p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-white text-xl font-bold">
          LawAI
        </Link>
        <div className="space-x-4">
          <Link href="/chat" className="text-white hover:text-gray-300">
            Try Out
          </Link>
          <Link href="/about" className="text-white hover:text-gray-300">
            About Us
          </Link>
          <Link href="/security" className="text-white hover:text-gray-300">
            Security
          </Link>
          <Link href="/login" className="text-white hover:text-gray-300">
            Login
          </Link>
        </div>
      </div>
    </nav>
  );
}
