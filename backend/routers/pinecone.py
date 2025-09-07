from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
from pinecone import Pinecone

router = APIRouter(prefix="/pinecone")

# Initialize Pinecone once 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host = os.getenv("PINECONE_HOST"))
# Connect to index and namespace

#Query chunk namespace
"""
query: query of the text 
top_k: top k results to return
"""
@router.post("/query-chunks/")
async def query_pinecone(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query")
        #if no query , return exception
        if not query_text:
            return JSONResponse(content={"error": "Query text is required"}, status_code=400)
        top_k = int(data.get("top_k", 10))

        results = index.search(
            namespace="chunks",
            query={
                "top_k": top_k,
                "inputs": {"text": query_text}
            },
            fields=["text", "judgment_date", "name", "uri"]
        )

        return JSONResponse(content=results.to_dict())
    except Exception as e:
        print("Pinecone error:", e)
        return JSONResponse(content={"error": "Failed to query Pinecone"}, status_code=500)


#Query summary namespace
"""
query: query of the text 
top_k: top k results to return
"""
@router.post("/query-summary/")
async def query_pinecone(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query")
        if not query_text:
            return JSONResponse(content={"error": "Query text is required"}, status_code=400)
        top_k = int(data.get("top_k", 10))


        results = index.search(
            namespace="summary",
            query={
                "top_k": top_k,
                "inputs": {"text": query_text}
            }
        )

        return JSONResponse(content=results.to_dict())
    except Exception as e:
        print("Pinecone error:", e)
        return JSONResponse(content={"error": "Failed to query Pinecone"}, status_code=500)
