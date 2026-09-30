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

app = FastAPI(title="Official AI Sales Agent")

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

SYSTEM_PROMPT = """
You are the Official AI Sales & Consultation Executive representing NextGen Sol.
Your goals:
1. Welcome visitors warmly and professionally.
2. Answer queries concisely about our premium digital and tech solutions.
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

# Cache the best working model dynamically
CACHED_MODEL = None

def get_active_model(client: genai.Client) -> str:
    global CACHED_MODEL
    if CACHED_MODEL:
        return CACHED_MODEL

    try:
        # Google API se automatically active models ki list maango
        models = client.models.list()
        candidates = []
        for m in models:
            name = m.name
            # Pro models free tier par allow nahi hotay (quota 0 hota hai), sirf Flash / Standard models lein
            if "flash" in name.lower() and "generateContent" in getattr(m, "supported_actions", ["generateContent"]):
                candidates.append(name.replace("models/", ""))

        if candidates:
            # Sab se latest active model select karo
            CACHED_MODEL = candidates[-1]
            return CACHED_MODEL
    except Exception as e:
        print(f"Auto-detection fallback warning: {e}")

    # Fallback to general flash if listing fails
    return "gemini-3.8-flash"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return JSONResponse({"reply": "GEMINI_API_KEY environment variable set nahi hai."}, status_code=500)

        client = genai.Client(api_key=api_key)

        prompt_text = f"System Instruction: {SYSTEM_PROMPT}\n\n"
        for msg in payload.history[-6:]:
            prompt_text += f"{msg.role.capitalize()}: {msg.content}\n"
        prompt_text += f"User: {payload.message}\nAssistant:"

        # Google se live valid model uthao
        selected_model = get_active_model(client)

        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt_text
            )
            raw_text = response.text or "I apologize, could you please repeat that?"
        except Exception as api_err:
            # Agar rate limit (429) ya issue aaye to cached reset kar ke ek dafa fallback try karein
            global CACHED_MODEL
            CACHED_MODEL = None
            return JSONResponse({"reply": f"Model busy: {str(api_err)}"}, status_code=500)

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
                    requirement=lead_json.get("requirement", "Interested in consultation")
                )
            except Exception as err:
                print(f"Error parsing lead data: {err}")
            return JSONResponse({"reply": clean_reply, "lead_captured": True})

        return JSONResponse({"reply": raw_text, "lead_captured": False})

    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"}, status_code=500)

@app.get("/leads", response_class=HTMLResponse)
async def view_leads(request: Request):
    db_path = BASE_DIR / "database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, requirement, status, created_at FROM leads ORDER BY id DESC")
    leads = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse(request=request, name="leads.html", context={"leads": leads})
