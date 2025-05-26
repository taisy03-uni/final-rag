"use client";

import { usePathname } from 'next/navigation';
import Navbar from './Navbar';

export default function NavbarWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isChat = pathname === '/chat';

  return (
    <>
      {!isChat && <Navbar />}
      <main className={isChat ? '' : 'container mx-auto p-4'}>{children}</main>
    </>
  );
} 