from collections import defaultdict
from typing import List, Dict

conversations: Dict[str, List] = defaultdict(list)

def get_history(session_id: str) -> List:
    return conversations[session_id]

def add_message(session_id: str, role: str, content: str):
    conversations[session_id].append({
        "role": role,
        "content": content
    })

def clear_history(session_id: str):
    conversations[session_id] = []

def build_prompt_with_history(context: str, question: str, session_id: str) -> str:
    history = get_history(session_id)
    
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If answer is not in context, say I don't know based on the documents.

Context:
{context}

Previous conversation:
{history_text}
User: {question}
Assistant:"""
    
    return prompt
