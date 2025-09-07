from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.encoders import jsonable_encoder
from .pinecone import query_pinecone_chunks

load_dotenv()  # Load environment variables from .env

router = APIRouter(prefix="/chatgpt")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/queryai/")
async def get_model_response(prompt: str, reasoning:str = "medium", instructions:str = "You are a helpfull legal AI assistant." , model: str = "gpt-5-mini") -> str:
    response = client.responses.create(
        model=model,
        input= prompt,
        reasoning={"effort": reasoning},
        instructions=instructions,
    )
    #return JSONResponse(content=jsonable_encoder(response))
    return response.output_text

# Helper function to triage the user query
@router.post("/query_triage/")
async def triage_query(context, query: str) -> str:
    prompt = f"""
    You are an AI assistant tasked with triaging user queries. 
    Classify the following query into one of three categories ONLY:
    1. GENERAL_QUESTION - general legal questions not requiring case law.
    2. ABOUT_ME - questions about you, your capabilities, or general conversation starters like "hey".
    3. CASELAW_QUESTION - user is asking about specific cases or judgments, or has provided a scenario of a case. Assume most scenarios will require case law.
    4. OUT_OF_SCOPE - questions unrelated to law or legal advice.

    History: {context}

    Query: "{query}"

    Return only one label from the three above. Do not return any explanations or additional text. Return the label bassed on the query, the history is there to give you more context about the user.
    """
    label = await get_model_response(prompt, reasoning="low")
    return label

# Helper function to answer general questions
@router.post("/query_general/")
async def answer_general_question(context, query: str, language: str) -> str:
    language_instruction = (
        "Please respond in British English using UK legal terminology."
        if language.lower() == "british"
        else "Please respond in American English using US legal terminology."
    )
    prompt = f"""
    You are a helpfull legal AI assistant which assists with case law research. Given the previous context with the user below, provide a concise and accurate answer to the user's question.
    History: {context}
    \n 
    {language_instruction}\n Answer this question: {query}
    
    """
    response = await get_model_response(prompt, reasoning="medium")
    return response

@router.post("/query_aboutme/")
async def answer_about_me_question(context, query: str, language: str) -> str:
    language_instruction = (
        "Please respond in British English using UK legal terminology."
        if language.lower() == "british"
        else "Please respond in American English using US legal terminology."
    )
    prompt = f"""
    You are a helpfull legal AI assistant. Given the previous context with the user below, provide a concise and accurate answer to the user's question.
    \n
    Information about you:
    Your name: Lexi
    How you work: You scan your internal knowledge base and the provided context to find the most relevant information to answer the user's question. 
    Dataset: You hold a collection of 66,000 UK case law documents from 2000 to 2023, which you can reference to provide accurate legal information from the National Archives.
    Who created you: Taise Sosina, a AI student at Imperial College London, created you as part of her final year project to assist law students and professionals in accessing legal information quickly and accurately.
    Your favourite colour: Blue/Purple

    \n
    Previous Interactions with the user:
    History: {context}
    \n 
    {language_instruction}\n Answer this question: {query}
    
    """
    response = await get_model_response(prompt)
    return response

# Helper function to answer general questions
async def answer_outofscope() -> str:
    return "I'm sorry, but I am unable to assist with that request as it is out of my scope. I am here to help with legal questions and find case law only."

@router.post("/querycaselaw/")
async def answer_caselaw_question(context, query, language, pinecone_data) -> str:
    prompt = f"""
    You are a helpfull legal AI assistant. Given the previous context with the user below, provide a concise and accurate answer to the user's question.
    Context: {pinecone_data}   
    \n
    Answer the user's question based on the context above.
    """
    response = await get_model_response(prompt, reasoning="high")
    return response

# Main POST endpoint
@router.post("/query/")
async def chatgpt_query(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query")
        language = data.get("language", "british")
        history = data.get("history", "No prior history")

        if not query_text:
            return JSONResponse(content={"error": "No query provided"}, status_code=400)
        # Step 1: Triage the query
        label = await triage_query(context=history, query=query_text)

        i = 0
        while label not in ["GENERAL_QUESTION", "ABOUT_ME", "OUT_OF_SCOPE", "CASELAW_QUESTION"] and i < 3:
            query_text = f"The previous query was: '{query_text}'. The triage label you provided was '{label}', which is not one of the accepted labels. Please re-evaluate the query and provide one of the accepted labels: GENERAL_QUESTION, ABOUT_ME, OUT_OF_SCOPE, CASELAW_QUESTION. Remember to return only the label without any additional text."
            label = await triage_query(context=history, query=query_text)
            i += 1
        
        # Step 2 & 3: Decide next action
        if label == "GENERAL_QUESTION":
            answer = await answer_general_question(context=history, query=query_text, language=language)
            return JSONResponse(content={"type": label, "answer": answer})
        elif label == "ABOUT_ME":
            answer = await answer_about_me_question(context=history, query= query_text, language=language)
            return JSONResponse(content={"type": label, "answer": answer})
        elif label == "OUT_OF_SCOPE":
            answer = await answer_outofscope()
            return JSONResponse(content={"type": label, "answer": answer})
        elif label == "CASELAW_QUESTION":
            pinecone_response = await query_pinecone_chunks(request)

            #answer = await answer_caselaw_question(context=history, query=query_text, language=language, pinecone_data=pinecone_response)
            return {"type": label, "answer": pinecone_response}
        else:
            return JSONResponse(content={"type": label, "answer": "Label not recognized after multiple attempts."})

    except Exception as e:
        print("ChatGPT router error:", e)
        return JSONResponse(content={"error": "Failed to process request"}, status_code=500)
