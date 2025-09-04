from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import pdf

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
