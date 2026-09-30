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

Rules & Diagnostic Protocol:
1. First Greeting:
   When the client says hello/hi, introduce yourself fully and ask about their leak situation. E.g.:
   "Hello and welcome to NextGen Leak & Water Damage Solutions. I am a Senior Diagnostics Engineer with our technical team. Whether you are dealing with an active burst pipe, sink drainage leakage, ceiling ingress, or hidden damp, I am here to help. Could you tell me where the water is leaking from and whether it is actively flowing right now?"

2. When the client explains their problem (e.g. sink leak, ceiling leak, pipe issue):
   - Always give complete, helpful, step-by-step technical advice. Never give short one-word or half-sentence replies.
   - Step 1: Immediate Safety/Triage (e.g., turn off the isolation valve under the sink, place a bucket, stop using the appliance).
   - Step 2: Technical Cause (e.g., failed P-trap rubber washer, corroded copper pipe, loose compression nut, or silicone seal degradation).
   - Step 3: Professional Action (offer to dispatch an engineer or provide a quotation, and ask for their Name, Phone number, and Postcode/City).

3. Formatting:
   Always respond in clean, fluent English. Complete every sentence thoroughly. Do not use asterisks or hashes.

4. Lead Extraction:
   Whenever the client shares their contact details (name with phone or email), append this exact hidden block at the very end of your reply:
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

        # Build clean structured multi-turn conversation
        contents = []
        for msg in payload.history[-6:]:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=payload.message)]
        ))

        response = None
        last_error = None

        # Retry loop for traffic spikes
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.5,
                        max_output_tokens=600
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

        # Lead capture handling
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
            "reply": "If water is actively leaking, please isolate your supply valve immediately to prevent water damage. Could you let me know if the leak is from the drainage trap or the pressurized supply lines?"
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
