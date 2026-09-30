import os
import json
import time
import sqlite3
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Water Leakage Specialist AI")

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

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

SYSTEM_PROMPT = """You are the Senior Technical Specialist at NextGen Leak & Water Damage Solutions.
Rules:
1. Always respond in plain, clean English. Never use markdown symbols like **, ###, or bullet asterisks.
2. Keep responses fast, concise, and reassuring (maximum 2 to 3 sentences).
3. Provide immediate safety triage first (such as shutting off the main stopcock or avoiding electrical hazards), then ask where the leak or dampness is located.
4. When the user provides contact details, append this exact block at the very end:
LEAD_DATA: {"name": "...", "email": "...", "phone": "...", "requirement": "..."}"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    history: List[ChatMessage]
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return JSONResponse({"reply": "System configuration error: GEMINI_API_KEY is missing on Render."}, status_code=500)

        client = genai.Client(api_key=api_key)

        contents = []
        for msg in payload.history[-4:]:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg.content)]
            ))
        
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=payload.message)]
        ))

        response = None
        last_error = None
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.4,
                        max_output_tokens=150
                    )
                )
                if response and response.text:
                    break
            except Exception as err:
                last_error = err
                time.sleep(1)

        if not response or not response.text:
            raise last_error

        raw_text = response.text.strip()

        if "LEAD_DATA:" in raw_text:
            parts = raw_text.split("LEAD_DATA:")
            clean_reply = parts[0].strip()
            try:
                lead_json = json.loads(parts[1].strip())
                save_lead(
                    name=lead_json.get("name", "N/A"),
                    email=lead_json.get("email", "N/A"),
                    phone=lead_json.get("phone", "N/A"),
                    requirement=lead_json.get("requirement", "Water Leakage Consultation")
                )
            except Exception as parse_err:
                print(f"Lead parsing error: {parse_err}")
            return JSONResponse({"reply": clean_reply, "lead_captured": True})

        return JSONResponse({"reply": raw_text, "lead_captured": False})

    except Exception as e:
        print(f"Execution Error: {str(e)}")
        return JSONResponse({
            "reply": "Please shut off your main stopcock immediately if water is active. Where is the water leaking from so I can advise next steps?"
        }, status_code=200)

@app.get("/leads", response_class=HTMLResponse)
async def view_leads(request: Request):
    db_path = BASE_DIR / "database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, requirement, status, created_at FROM leads ORDER BY id DESC")
    leads = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse(request=request, name="leads.html", context={"leads": leads})
