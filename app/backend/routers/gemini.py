from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os

router = APIRouter(prefix="/gemini")

