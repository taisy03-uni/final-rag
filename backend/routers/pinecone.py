from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from pinecone import Pinecone

router = APIRouter(prefix="/pinecone")

load_dotenv() # Load environment variables from .env file   
#check if PINECONE_API_KEY is set
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable not set")

# Initialize Pinecone once (on startup)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host = os.getenv("PINECONE_HOST"))
# Connect to index and namespace


@router.post("/query-chunks/")
async def query_pinecone(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query")

        results = index.search(
            namespace="chunks",
            query={
                "top_k": 10,
                "inputs": {"text": query_text}
            },
            fields=["text", "judgment_date", "name", "uri"]
        )

        return JSONResponse(content=results.to_dict())
    except Exception as e:
        print("Pinecone error:", e)
        return JSONResponse(content={"error": "Failed to query Pinecone"}, status_code=500)

@router.post("/query-summary/")
async def query_pinecone(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query")

        results = index.search(
            namespace="summary",
            query={
                "top_k": 10,
                "inputs": {"text": query_text}
            },
            fields=["text", "judgment_date", "name", "uri"]
        )

        return JSONResponse(content=results.to_dict())
    except Exception as e:
        print("Pinecone error:", e)
        return JSONResponse(content={"error": "Failed to query Pinecone"}, status_code=500)
