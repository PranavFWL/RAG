from pydantic import BaseModel
from typing import List, Optional

# This is the shape of REQUEST (what user sends TO our API)
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.2

# This is the shape of one SOURCE document returned
class Source(BaseModel):
    source: str
    page: str
    score: float
    preview: str

# This is the shape of RESPONSE (what our API sends BACK)
class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.2
