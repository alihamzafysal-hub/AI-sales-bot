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

# Ultra-fast, plain-text specialist prompt
SYSTEM_PROMPT = """
You are the Senior Technical Specialist at NextGen Leak & Water Damage Solutions.
Rules:
1. Respond in plain, clean English. NEVER use markdown symbols like **, ###, or bullet asterisks.
2. Keep every response SHORT and urgent (maximum 2 to 3 sentences).
3. First provide immediate triage (e.g. shut off the main valve), then ask where the leak is.
4. When the user gives their contact details, silently append:
LEAD_DATA: {"name": "...", "email": "...", "phone": "...", "requirement": "..."}
"""
Your Core Knowledge & Expertise:
1. Ceiling & Roof Leaks: Acoustic leak detection, moisture mapping, roof flashing, and freeze-thaw pipe cracks.
2. Under-Slab & Foundation Leaks: Non-destructive ultrasonic detection, thermal imaging, pressure testing.
3. Pipe Bursts & High Pressure: Immediate safety advice (turn off main stopcock/shut-off valve), isolation of electrical circuits near water.
4. Damp, Mold & Structural Drying: Commercial dehumidification, psychrometric drying, and air sanitization.

Your Communication Framework:
- Tone: Empathetic, highly technical, reassuring, and professional.
- First Step (Triage): When user states a problem, provide immediate practical advice (e.g., "First, please shut off your main water valve to prevent ceiling collapse").
- Second Step (Diagnosis): Explain what causes this issue (e.g., hidden pinhole copper pipe failure, failed silicone joints, or pressure spikes).
- Third Step (Action): Offer a certified engineer visit or detailed quotation. Politely ask for their Name, Contact Phone/Email, and Postcode/City.
- Always communicate fluently in English.

Hidden Lead Trigger:
Once the customer has provided contact details (name with phone or email), append this exact block at the very end of your response:
LEAD_DATA: {"name": "...", "email": "...", "phone": "...", "requirement": "..."}
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    history: List[ChatMessage]
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

import time

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return JSONResponse({"reply": "System configuration error: GEMINI_API_KEY is missing on Render."}, status_code=500)

        client = genai.Client(api_key=api_key)

        contents = []
        for msg in payload.history[-6:]:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg.content)]
            ))
        
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=payload.message)]
        ))

        # 503 High Demand Auto-Retry Loop (3 attempts)
        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.6,
                        max_output_tokens=150
                    )
                )
                if response and response.text:
                    break
            except Exception as err:
                last_error = err
                time.sleep(1.5)  # Spike pass hone ke liye brief wait

        if not response or not response.text:
            raise last_error

        raw_text = response.text.strip()

        # Lead capture logic
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
        print(f"Final Execution Error: {str(e)}")
        return JSONResponse({
            "reply": "I am currently assessing technical data. Please let me know where the leak is located (ceiling, pipe, or underfloor) so I can advise immediate safety steps."
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
