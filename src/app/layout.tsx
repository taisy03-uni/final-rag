import type { Metadata } from "next";
import { Puritan } from "next/font/google";
import './globals.css';
import NavbarWrapper from '@/components/NavbarWrapper';

// Configure Puritan font
const puritan = Puritan({
  weight: ["400", "700"],
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-puritan",
});

export const metadata: Metadata = {
  title: "LADA.AI - Legal Assistant",
  description: "Your advanced AI legal assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${puritan.variable}`}>
      <body className="font-sans antialiased">
        <NavbarWrapper>{children}</NavbarWrapper>
      </body>
    </html>
  );
}