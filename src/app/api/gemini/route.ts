import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from "@google/generative-ai";

export const runtime = 'edge';

type ChatMessage = {
  text: string;
  isOutgoing: boolean;
};

type RequestBody = {
  messages: ChatMessage[];
  isFirstMessage?: boolean;
};

export async function POST(req: Request) {
  try {
    const { messages, isFirstMessage }: RequestBody = await req.json();
    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    // Get the latest user message
    const userQuery = messages[messages.length - 1].text;

    // Get base URL from request for absolute URL
    const baseUrl = new URL(req.url).origin;

    // Call Pinecone API route
    const pineconeResponse = await fetch(`${baseUrl}/api/pinecone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: userQuery }),
    });

    if (!pineconeResponse.ok) {
      throw new Error('Failed to query Pinecone');
    }

    // Parse and handle Pinecone response safely
    const pineconeData = await pineconeResponse.json();
    
    const legalContext = JSON.stringify(pineconeData, null, 2);
    // Create complete prompt template
    const prompt = `
      You are a legal research assistant specialized in UK law. Below is relevant legal context followed by the user's question.

      Legal Context:
      ${legalContext}

      User Question:
      ${userQuery}

      Please provide a thorough, professional answer based on the legal context provided. 
      If the context doesn't fully answer the question:
      1. Clearly indicate this
      2. Provide general legal guidance based on your knowledge
      3. Suggest potential next steps
      4. Always cite sources where possible
      
      Structure your response with:
      - Clear headings
      - Concise explanations
      - Relevant case law references
      - Practical advice
    `;

    // Handle first message specially
    if (isFirstMessage) {
      const result = await model.generateContent(prompt);
      const response = await result.response;
      return NextResponse.json({ response: response.text() + `\n\nLegal Context:\n${legalContext}`});
    }

    // Normal conversation flow
    const chat = model.startChat({
      history: messages.slice(0, -1).map((msg) => ({
        role: msg.isOutgoing ? "user" : "model",
        parts: [{ text: msg.text }],
      })),
    });

    const result = await chat.sendMessage(prompt);
    const response = await result.response;
    // add Legal Context to the response

    return NextResponse.json({ response: response.text() + `\n\nLegal Context:\n${legalContext}` });

  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat message' },
      { status: 500 }
    );
  }
}