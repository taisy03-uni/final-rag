from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import pdf, pinecone, gemini, chatgpt

app = FastAPI()

# Allow frontend requests (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf.router)
app.include_router(pinecone.router)  
app.include_router(chatgpt.router)  
app.include_router(gemini.router)