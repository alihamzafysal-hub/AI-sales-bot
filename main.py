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

SYSTEM_PROMPT = """You are the Senior Technical Diagnostics Specialist at NextGen Leak & Water Damage Solutions.

Rules & Communication Behavior:
1. When a client initiates contact with a greeting (e.g., 'hi', 'hello', 'hey', 'start'), ALWAYS respond with this exact comprehensive technical introduction:
"Hello, and welcome to NextGen Leak & Water Damage Solutions. I’m a Senior Diagnostics Engineer with the technical team. Whether you're dealing with an active emergency leak, an unexplained drop in system pressure, hidden damp, or water ingress through ceilings or foundations, I'm here to help you assess and resolve the issue safely. To help me give you the best advice: Are you currently experiencing an active leak or water damage? If so, could you briefly describe what you're seeing?"

2. When the client explains their specific problem:
   - Provide immediate safety triage advice first (e.g., isolate main stopcock/shut-off valve, avoid electrical switches in wet areas, or place collection buckets).
   - Offer a technical diagnosis of the root cause (e.g., pinhole copper pipe corrosion, failed waste trap seals, high system pressure, or external masonry water ingress).
   - Offer dispatching a certified acoustic detection engineer or damp survey team. Politely request their Name, Contact Phone Number, and Postcode/City to arrange immediate support.

3. Always communicate in professional, clear, and reassuring English. Never truncate sentences.

4. Once the client provides their contact details (name and phone/email), silently append this exact block at the very end of your response:
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
            return JSONResponse({"reply": "Configuration Notice: GEMINI_API_KEY is missing on Render."}, status_code=500)

        client = genai.Client(api_key=api_key)

        prompt_text = f"System Instructions:\n{SYSTEM_PROMPT}\n\nRecent Conversation History:\n"
        for msg in payload.history[-6:]:
            role_label = "Client" if msg.role == "user" else "Specialist"
            prompt_text += f"{role_label}: {msg.content}\n"
        prompt_text += f"Client: {payload.message}\nSpecialist:"

        response = None
        last_error = None
        
        # 2-attempt retry loop for high traffic spikes
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
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

        # Handle lead capture
        if "LEAD_DATA:" in raw_text:
            parts = raw_text.split("LEAD_DATA:")
            clean_reply = parts[0].strip()
            try:
                lead_json = json.loads(parts[1].strip())
                save_lead(
                    name=lead_json.get("name", "N/A"),
                    email=lead_json.get("email", "N/A"),
                    phone=lead_json.get("phone", "N/A"),
                    requirement=lead_json.get("requirement", "Water Leakage Technical Consultation")
                )
            except Exception as parse_err:
                print(f"Lead parsing error: {parse_err}")
            return JSONResponse({"reply": clean_reply, "lead_captured": True})

        return JSONResponse({"reply": raw_text, "lead_captured": False})

    except Exception as e:
        print(f"Execution Error: {str(e)}")
        return JSONResponse({
            "reply": "If you are dealing with an active water leak, please locate and shut off your main stopcock valve immediately. Where is the water coming from (ceiling, pipes, or underfloor) so I can advise immediate safety steps?"
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
