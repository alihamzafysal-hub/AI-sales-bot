import os
import json
import sqlite3
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Official AI Sales Agent")

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Gemini API Client
api_key = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None

def init_db():
    db_path = BASE_DIR / "database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            requirement TEXT,
            status TEXT DEFAULT 'New',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_lead(name: str, email: str, phone: str, requirement: str):
    db_path = BASE_DIR / "database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO leads (name, email, phone, requirement) VALUES (?, ?, ?, ?)",
        (name, email, phone, requirement)
    )
    conn.commit()
    conn.close()

SYSTEM_PROMPT = """
You are the Official AI Sales & Consultation Executive representing the company.
Your goals:
1. Welcome visitors warmly and professionally.
2. Answer queries concisely about our premium services/solutions.
3. Politely collect their Name, Business Email/Phone, and project requirements.
4. Once you have acquired contact details, append this exact hidden block at the very end of your response:
LEAD_DATA: {"name": "...", "email": "...", "phone": "...", "requirement": "..."}

Maintain an elegant, helpful, and executive tone at all times.
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    history: List[ChatMessage]
    message: str

# FIX: Yahan 'request=request' pehle pass kiya hai taake naye Starlette mein crash na ho
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    try:
        if not client:
            return JSONResponse({"reply": "GEMINI_API_KEY environment variable mein set nahi hai."}, status_code=500)

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=SYSTEM_PROMPT)])]
        for msg in payload.history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=payload.message)]))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        raw_text = response.text or "I apologize, could you please repeat that?"

        if "LEAD_DATA:" in raw_text:
            parts = raw_text.split("LEAD_DATA:")
            clean_reply = parts[0].strip()
            try:
                lead_json = json.loads(parts[1].strip())
                save_lead(
                    name=lead_json.get("name", "N/A"),
                    email=lead_json.get("email", "N/A"),
                    phone=lead_json.get("phone", "N/A"),
                    requirement=lead_json.get("requirement", "Interested in consultation")
                )
            except Exception as e:
                print(f"Error parsing lead data: {e}")
            return JSONResponse({"reply": clean_reply, "lead_captured": True})

        return JSONResponse({"reply": raw_text, "lead_captured": False})

    except Exception as e:
        return JSONResponse({"reply": "System busy, please try again shortly.", "error": str(e)}, status_code=500)

# FIX: Yahan bhi 'request=request' pehle aur context alag se pass kiya hai
@app.get("/leads", response_class=HTMLResponse)
async def view_leads(request: Request):
    db_path = BASE_DIR / "database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, requirement, status, created_at FROM leads ORDER BY id DESC")
    leads = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse(request=request, name="leads.html", context={"leads": leads})
