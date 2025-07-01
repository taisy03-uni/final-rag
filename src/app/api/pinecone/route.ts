import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { query } = await req.json();
    
    const pinecone = new Pinecone({ 
      apiKey: process.env.PINECONE_API_KEY! 
    });
    
    const index = pinecone.index("chunks", process.env.PINECONE_HOST).namespace("chunks");

    const results = await index.searchRecords({
      query: {
        topK: 10,
        inputs: { text: query },
      },
      fields: ['text', 'judgment_date', 'name', "uri"],
    });

    return NextResponse.json(results);
  } catch (error) {
    console.error('Pinecone error:', error);
    return NextResponse.json(
      { error: 'Failed to query Pinecone' },
      { status: 500 }
    );
  }
}