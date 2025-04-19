import type { Metadata } from "next";
import { Puritan } from "next/font/google";
import './globals.css';
import Navbar from '@/components/Navbar';

// Configure Puritan font
const puritan = Puritan({
  weight: ["400", "700"],
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-puritan",
});

export const metadata: Metadata = {
  title: "My App",
  description: "Using Puritan from Google Fonts",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${puritan.variable}`}>
      <body className="font-sans antialiased">
        <Navbar />
        <main className="container mx-auto p-4">{children}</main>
      </body>
    </html>
  );
}