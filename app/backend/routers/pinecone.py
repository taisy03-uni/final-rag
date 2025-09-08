from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
from pinecone import Pinecone
from .chatgpt import get_model_response
from support.metadata import MetadataEnhancer

router = APIRouter(prefix="/pinecone")

# Initialize Pinecone once 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host = os.getenv("PINECONE_HOST"))
# Connect to index and namespace


enhancer = MetadataEnhancer()


# Helper function to answer case law questions
async def metadata(query: str) -> str:
    #expand query with metadata
    #TODO
    return None

async def rerank(context, query: str) -> str:
    #TODO
    return None


#Query chunk namespace
"""
query: query of the text 
top_k: top k results to return
"""

# Internal function
async def query_chunks_internal(query_text: str, top_k: int = 10) -> dict:
    try:
        results = index.search(
            namespace="chunks",
            query={
                "top_k": top_k,
                "inputs": {"text": query_text}
            },
            fields=["text", "judgment_date", "name", "uri"]
        )
        return results.to_dict()
    except Exception as e:
        print("Pinecone error:", e)
        return {"error": "Failed to query Pinecone"}


@router.post("/query-chunks/")
async def query_pinecone_chunks(request: Request):
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
async def query_pinecone_summary(request: Request):
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

@router.post("/find-summary/")
async def find_pinecone_summary(request: Request):
    try:
        data = await request.json()
        uri = data.get("uri")
        if not uri:
            return JSONResponse(content={"error": "Query text is required"}, status_code=400)
        id = [i for i in index.list(prefix=uri, limit=100, namespace="summary")][0]
        #list of list to flat list

        print("Found summary IDs:", id)
        results = index.fetch(ids=id, namespace="summary")
        # Convert vectors to dict of ID → metadata
        summaries = {
            _id: vec.metadata
            for _id, vec in results.vectors.items()
        }
        #sort summaries by key where uri_1 is first uri_2... and so on
        summaries = dict(sorted(summaries.items(), key=lambda item: item[0]))
        #combinae all item["summary"] into one string
        whole_summary = " ".join([item["text"] for item in summaries.values() if "text" in item])

        return JSONResponse(content={"summary": whole_summary})
    except Exception as e:
        print("Pinecone error:", e)
        return JSONResponse(content={"error": "Failed to query Pinecone"}, status_code=500)


@router.post("/query-metadata/")
async def metadata_guided_search(request: Request):
        data = await request.json()
        query = data.get("query")
        top_k = int(data.get("top_k", 10))
        metadata = await enhancer.extract_metadata_with_llm(query)
        """Perform metadata-guided search with multiple strategies"""
        enhanced_queries = enhancer.enhance_query_with_metadata(query, metadata)
        
        all_results = []
        seen_uris = set()
        
        # Weight queries based on metadata confidence and type
        query_weights = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6]  # Decreasing weights
        
        for i, enhanced_query in enumerate(enhanced_queries):
            try:
                results = await query_chunks_internal(query_text=enhanced_query,top_k=3)
                weight = query_weights[i] if i < len(query_weights) else 0.5
                
                for hit in results['result']['hits']:
                    uri = hit['fields']['uri']
                    if uri not in seen_uris:
                        hit['_score'] = hit['_score'] * weight
                        hit['metadata_enhanced'] = True
                        all_results.append(hit)
                        seen_uris.add(uri)
                        
            except Exception as e:
                print("Error:", e)
                continue
        
        # Sort by weighted scores
        all_results.sort(key=lambda x: x['_score'], reverse=True)
        
        return {'result': {'hits': all_results[:top_k]}}