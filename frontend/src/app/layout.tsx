import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocuAI | Document Intelligence",
  description: "Chat with multi-file PDFs using production RAG powered by Llama 3.1 8B, Qdrant Vector Cloud, and Google Auth.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full antialiased font-sans"
    >
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900 selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
