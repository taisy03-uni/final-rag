from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import pinecone  # make sure to install: pip install pinecone-client

app = FastAPI()

# Initialize Pinecone once (on startup)
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))  # environment is required

index_name = "chunks"
namespace = "chunks"

@app.post("/query-pinecone/")
async def query_pinecone(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query")

        # Connect to the index
        index = pinecone.Index(index_name)

        # Perform search
        results = index.query(
            top_k=10,
            include_metadata=True,
            filter=None,
            namespace=namespace,
            query=query_text  # For text search, you may need to embed first
        )

        return JSONResponse(content=results)
    except Exception as e:
        print("Pinecone error:", e)
        return JSONResponse(content={"error": "Failed to query Pinecone"}, status_code=500)
