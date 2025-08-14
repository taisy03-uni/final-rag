import { NextResponse } from 'next/server';
import OpenAI from 'openai';

export const runtime = 'edge';

type ChatMessage = {
  text: string;
  isOutgoing: boolean;
};

type RequestBody = {
  messages: ChatMessage[];
  currentLanguage: string;
};

function evaluateResponse(responseText: string, pineconeData: any, searchResults: any): string {
  // Extract case names from response text
  const caseNames = responseText.match(/Case Name: ([^\n]+)/g) || [];
  const caseNameSet = new Set(caseNames.map(name => 
    name.replace('Case Name: ', '').replace('**', '').trim()
  ));

  // Get case names from Pinecone data
  const pineconeCases = pineconeData.result.hits.map((hit: any) => hit.fields.name);
  const pineconeCaseSet = new Set(pineconeCases);
  
  // Count matches between response and Pinecone cases
  let relevantCount = 0;
  caseNameSet.forEach(name => {
    if (pineconeCaseSet.has(name)) {
      relevantCount++;
    }
  });

  // Extract search result insights
  const searchSummary = searchResults ? `
    OpenAI Search Results:
    - Found ${searchResults.length} relevant search results
    - Search sources included web data and knowledge base
  ` : '';

  const evaluationReport = `
    Evaluation Report:
    =================
    Relevant cases found: ${relevantCount}
    Cases extracted from Pinecone: ${pineconeCaseSet.size}
    Cases mentioned by AI: ${caseNameSet.size}
    ${searchSummary}
    =================
    Matching cases: ${Array.from(caseNameSet)
      .filter(name => pineconeCaseSet.has(name))
      .join(', ') || 'None'}
  `;

  return evaluationReport;
}

async function performOpenAISearch(client: OpenAI, query: string): Promise<any> {
  try {
    // Use OpenAI's chat completion with search-like functionality
    // This simulates search by using the model to find relevant information
    const searchCompletion = await client.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: "You are a legal research assistant. Search for and provide relevant UK legal cases related to the user's query. Format your response as structured search results."
        },
        {
          role: "user", 
          content: `Scenario: ${query} Output: Please provide relevant UK case law in the following json format:
                    [{
                    case_name: "Case Name",
                    summary: "Brief summary of the case",
                    citation: "Citation details",
                    date: "Judgment date",
                    },...]`
        }
      ],
      max_tokens: 500,
      temperature: 0.1
    });

    return {
      results: searchCompletion.choices[0]?.message?.content || "No search results found",
      usage: searchCompletion.usage
    };
  } catch (error) {
    console.error('OpenAI search error:', error);
    return {
      results: "Search functionality temporarily unavailable",
      usage: null
    };
  }
}

export async function POST(req: Request) {
  try {
    const { messages, currentLanguage }: RequestBody = await req.json();
    
    // Initialize OpenAI client
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    // Get the latest user message
    const userQuery = messages[messages.length - 1].text;

    // Get base URL from request for absolute URL
    const baseUrl = new URL(req.url).origin;

    // Perform OpenAI search first
    const searchResults = await performOpenAISearch(openai, userQuery);

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
    
    // Create complete prompt template with search results and language consideration
    const prompt = `
      You are a legal research assistant specialized in UK law. ${languageInstruction}
      Below is relevant legal context from our database, search results, and the user's question.

      OpenAI Search Results:
      ${searchResults.results}

      Legal Context from Database:
      ${legalContext}

      User Case:
      ${userQuery}

      Given the scenario provided by the user, the search results, and the relevant legal context, please provide a detailed and accurate response which states:
      1. The relevant UK case law - provide the name of the cases and a brief summary of how it relates to the user case
      2. Additional insights from the search results that may be relevant
      3. Sort the cases by relevance to the user case
      4. Each case should be formatted as follows: Case Name: [Case Name]
    `;

    // Get OpenAI chat completion
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        ...messages.slice(0, -1).map((msg) => ({
          role: msg.isOutgoing ? "user" as const : "assistant" as const,
          content: msg.text,
        })),
        {
          role: "user",
          content: prompt,
        }
      ],
      max_tokens: 2000,
      temperature: 0.7,
    });

    const responseText = completion.choices[0]?.message?.content || "No response generated";
    console.log(`OpenAI response: ${responseText}`);
    
    const evaluation = evaluateResponse(responseText, pineconeData, searchResults);
    console.log(`Evaluation: ${evaluation}`);
    
    const fullResponse = `${searchResults.results}\n\n${responseText}`;
    
    return NextResponse.json({ response: fullResponse });

  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat message' },
      { status: 500 }
    );
  }
}