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
    const genAI = new GoogleGenerativeAI(process.env.API_KEY!);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    // Handle first message specially
    if (isFirstMessage) {
      const prompt = messages[0].text;
      const result = await model.generateContent(prompt);
      const response = await result.response;
      return NextResponse.json({ response: response.text() });
    }

    // Normal conversation flow
    const chat = model.startChat({
      history: messages.slice(0, -1).map((msg: ChatMessage) => ({
        role: msg.isOutgoing ? "user" : "model",
        parts: [{ text: msg.text }],
      })),
    });

    const result = await chat.sendMessage(messages[messages.length - 1].text);
    const response = await result.response;
    return NextResponse.json({ response: response.text() });

  } catch (error) {
    console.error('Gemini API error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat message' },
      { status: 500 }
    );
  }
}