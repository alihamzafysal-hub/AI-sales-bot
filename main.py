import os
import json
import sqlite3
from datetime import datetime
from typing import List
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Official AI Sales Agent")
templates = Jinja2Templates(directory="templates")

# Gemini Client (Set your GEMINI_API_KEY environment variable)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

# ----------------- DATABASE SETUP -----------------
def init_db():
    conn = sqlite3.connect("database.db")
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
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO leads (name, email, phone, requirement) VALUES (?, ?, ?, ?)",
        (name, email, phone, requirement)
    )
    conn.commit()
    conn.close()

# ----------------- SYSTEM PROMPT -----------------
SYSTEM_PROMPT = """
You are the Official AI Sales & Consultation Executive representing the company.
Your goals:
1. Welcome visitors warmly and professionally.
2. Answer queries concisely about our premium services/solutions.
3. Politely collect their Name, Business Email/Phone, and project requirements.
4. Once you have acquired their contact details (or if they express explicit interest in booking a consultation), output a hidden JSON block at the very end of your response like this:
LEAD_DATA: {"name": "...", "email": "...", "phone": "...", "requirement": "..."}

Maintain an elegant, helpful, and executive tone at all times.
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    history: List[ChatMessage]
    message: str

# ----------------- ROUTES -----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    try:
        # Build prompt history
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=SYSTEM_PROMPT)])]
        
        for msg in payload.history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=payload.message)]))

        # Generate response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        raw_text = response.text or "I apologize, could you please repeat that?"

        # Extract lead if detected
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
        return JSONResponse({"reply": "Our consultation lines are active. Please leave your contact details or try again shortly.", "error": str(e)}, status_code=500)

@app.get("/leads", response_class=HTMLResponse)
async def view_leads(request: Request):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, requirement, status, created_at FROM leads ORDER BY id DESC")
    leads = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse("leads.html", {"request": request, "leads": leads})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)