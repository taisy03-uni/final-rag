import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from "@google/generative-ai";

export const runtime = 'edge';

type ChatMessage = {
  text: string;
  isOutgoing: boolean;
};

type RequestBody = {
  messages: ChatMessage[];
  currentLanguage: string;
};

function evaluateResponse(responseText: string, pineconeData: any): string {
  // Step 1: Extract case names from response text
  const caseNames = responseText.match(/Case Name: ([^\n]+)/g) || [];
  //remove ** and trim whitespace and also 

  const caseNameSet = new Set(caseNames.map(name => 
    name.replace('Case Name: ', '').replace('**', '').trim()
  ));

  // Step 2: Get case names from Pinecone data
  const pineconeCases = pineconeData.result.hits.map((hit: any) => hit.fields.name);
  const pineconeCaseSet = new Set(pineconeCases);
  
  // Step 3: Count matches between response and Pinecone cases
  let relevantCount = 0;
  caseNameSet.forEach(name => {
    if (pineconeCaseSet.has(name)) {
      relevantCount++;
    }
  });

  // Step 4: Prepare evaluation report
  const evaluationReport = `
    Evaluation Report:
    =================
    Relevant cases found: ${relevantCount}
    Cases extracted from Pinecone: ${pineconeCaseSet.size}
    names extracted from response: ${Array.from(pineconeCaseSet)}
    Cases mentioned by AI: ${caseNameSet.size}
    Cases in response: ${Array.from(caseNameSet)}
    =================
    Matching cases: ${Array.from(caseNameSet)
      .filter(name => pineconeCaseSet.has(name))
      .join(', ') || 'None'}
  `;

  return evaluationReport;
}

export async function POST(req: Request) {
  try {
    const { messages, currentLanguage}: RequestBody = await req.json();
    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    console.log(`Using model: ${model.model}`);
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

    const pineconeData = await pineconeResponse.json();

    // Format the language instruction based on selection
    const languageInstruction = currentLanguage === 'american' 
      ? "Please respond in American English using US legal terminology."
      : "Please respond in British English using UK legal terminology.";

    const legalContext = JSON.stringify(pineconeData, null, 2);
    
    // Create complete prompt template with language consideration
    const prompt = `
      You are a legal research assistant specialized in UK law. ${languageInstruction}
      Below is relevant legal context followed by the user's question.

      Legal Context:
      ${legalContext}

      User Case:
      ${userQuery}

      Given the scenario provided by the user and the relevant legal context, please provide a detailed and accurate response which states:
      1. The relevant UK case Law - provide the name of the cases and a brief summary of how it related to the user case. 
      2. Sort the cases by relevance to the user case. 
      3. Each case should be formatted as follows: Case Name: [Case Name] 
    `;

    // Normal conversation flow
    const chat = model.startChat({
      history: messages.slice(0, -1).map((msg) => ({
        role: msg.isOutgoing ? "user" : "model",
        parts: [{ text: msg.text }],
      })),
    });

    const result = await chat.sendMessage(prompt);
    console.log(`Chat response: ${result.response.text()}`);
    const response = await result.response;
    console.log(`Response text: ${response.text()}`);
    const evaluation = evaluateResponse(response.text(), pineconeData);
    console.log(`Evaluation: ${evaluation}`);
    const fullResponse = 
    `${evaluation}\n\n${response.text()}`;
    
    return NextResponse.json({ response: fullResponse });

  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat message' },
      { status: 500 }
    );
  }
}