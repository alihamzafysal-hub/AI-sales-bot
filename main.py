import os
import re
import json
import time
import secrets
import sqlite3
import traceback
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Water Leakage Specialist AI")

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Model name ko Render ke env variable se badla ja sakta hai
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Jawab ki lambai. Thinking ke tokens isi mein count hote hain, is liye 200 bohat kam hai
# (jawab beech mein kat jata hai ya khali aata hai). Speed ke liye thinking LOW rakhi hai.
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "800"))

# Agar pehla model fail ho to ye agle models try honge
MODEL_CHAIN = []
for _m in [GEMINI_MODEL, "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]:
    if _m and _m not in MODEL_CHAIN:
        MODEL_CHAIN.append(_m)

LAST_ERRORS = []  # /diagnostics page ke liye

# ---------------------------------------------------------------
# Leads page security (HTTP Basic Auth)
# Render mein ye do env variables zaroor set karein:
#   LEADS_USERNAME  (default: admin)
#   LEADS_PASSWORD  (zaroori)
# ---------------------------------------------------------------
security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    admin_user = os.environ.get("LEADS_USERNAME", "admin")
    admin_pass = os.environ.get("LEADS_PASSWORD", "")

    if not admin_pass:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Leads portal disabled: LEADS_PASSWORD is not set on the server.",
        )

    user_ok = secrets.compare_digest(credentials.username.encode("utf-8"), admin_user.encode("utf-8"))
    pass_ok = secrets.compare_digest(credentials.password.encode("utf-8"), admin_pass.encode("utf-8"))

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


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
   - Always build on what the client has already told you. Never repeat a question they have already answered.

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


def build_contents(history: List[ChatMessage], message: str):
    """Gemini ke liye clean multi-turn history banata hai."""
    recent = list(history[-6:])

    # Gemini conversation ka pehla message 'user' ka hona chahiye
    while recent and recent[0].role != "user":
        recent.pop(0)

    contents = []
    for msg in recent:
        role = "user" if msg.role == "user" else "model"
        if not msg.content or not msg.content.strip():
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    return contents


def make_config(model: str) -> types.GenerateContentConfig:
    kwargs = dict(system_instruction=SYSTEM_PROMPT, max_output_tokens=MAX_OUTPUT_TOKENS)
    if hasattr(types, "ThinkingConfig"):
        if "2.5" in model:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif "gemini-3" in model:
            # Gemini 3.x: temperature ignore hota hai, MINIMAL allowed nahi, LOW sab se tez hai
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
            except Exception as cfg_err:
                print(f"[Gemini] thinking_level not supported by installed SDK: {cfg_err}")
    else:
        kwargs["temperature"] = 0.5
    return types.GenerateContentConfig(**kwargs)


def generate_reply(client, contents) -> str:
    """Har model ko baari baari try karta hai. Sab fail hon to aakhri error raise karta hai."""
    global LAST_ERRORS
    errors = []

    for model in MODEL_CHAIN:
        for attempt in (1, 2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=make_config(model),
                )
                text = (response.text or "").strip() if response else ""
                if text:
                    LAST_ERRORS = []
                    return text
                errors.append(f"{model}: empty response")
                print(f"[Gemini] {model}: empty response")
                break
            except Exception as err:
                msg = f"{model}: {type(err).__name__}: {err}"
                errors.append(msg)
                print(f"[Gemini] {msg}")
                text_err = str(err)
                # Ye errors retry se theek nahi hote, seedha agle model par jao
                if any(k in text_err for k in ("404", "NOT_FOUND", "400", "INVALID_ARGUMENT", "403", "PERMISSION_DENIED", "API key")):
                    break
                time.sleep(1)

    LAST_ERRORS = errors
    raise RuntimeError(" | ".join(errors) or "Gemini failed without returning an error")


def clean_formatting(text: str) -> str:
    """Prompt ke mutabiq asterisks aur hashes hata deta hai."""
    return text.replace("*", "").replace("#", "").strip()


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
        contents = build_contents(payload.history, payload.message)
        raw_text = generate_reply(client, contents)

        # Lead capture handling
        if "LEAD_DATA:" in raw_text:
            parts = raw_text.split("LEAD_DATA:", 1)
            clean_reply = clean_formatting(parts[0])
            try:
                match = re.search(r"\{.*\}", parts[1], re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found after LEAD_DATA")
                lead_json = json.loads(match.group(0))
                save_lead(
                    name=lead_json.get("name", "N/A"),
                    email=lead_json.get("email", "N/A"),
                    phone=lead_json.get("phone", "N/A"),
                    requirement=lead_json.get("requirement", "Water Leakage Consultation")
                )
                return JSONResponse({"reply": clean_reply, "lead_captured": True})
            except Exception as parse_err:
                print(f"Lead parsing error: {parse_err}")
                return JSONResponse({"reply": clean_reply, "lead_captured": False})

        return JSONResponse({"reply": clean_formatting(raw_text), "lead_captured": False})

    except Exception as e:
        print(f"Execution Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return JSONResponse({
            "reply": (
                "I am sorry, our diagnostics system is temporarily busy. In the meantime, if water is actively "
                "leaking, please turn off the isolation valve for that supply line and place a container under "
                "the leak. Please share your name, phone number and city, and one of our engineers will contact "
                "you shortly."
            ),
            "lead_captured": False
        }, status_code=200)


@app.get("/diagnostics")
async def diagnostics(admin: str = Depends(require_admin)):
    """Password ke peeche: Gemini connection ka live test."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    result = {
        "api_key_set": bool(api_key),
        "api_key_preview": (api_key[:4] + "..." + api_key[-3:]) if len(api_key) > 8 else "too short / missing",
        "models_to_try": MODEL_CHAIN,
        "tests": [],
    }
    if not api_key:
        return JSONResponse(result)

    client = genai.Client(api_key=api_key)
    for model in MODEL_CHAIN:
        try:
            r = client.models.generate_content(
                model=model,
                contents="Reply with the single word OK",
                config=make_config(model),
            )
            result["tests"].append({"model": model, "status": "WORKING", "reply": (r.text or "")[:50]})
        except Exception as err:
            result["tests"].append({"model": model, "status": "FAILED", "error": f"{type(err).__name__}: {err}"[:400]})
    return JSONResponse(result)


@app.get("/leads", response_class=HTMLResponse)
async def view_leads(request: Request, admin: str = Depends(require_admin)):
    db_path = BASE_DIR / "database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, requirement, status, created_at FROM leads ORDER BY id DESC")
    leads = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse(request=request, name="leads.html", context={"leads": leads})
