from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import random
import time
import hashlib
import requests
from core.config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# In-memory session store & OTP store (target -> {code, expires_at})
otp_store = {}
users_db = {}

class OTPSendRequest(BaseModel):
    target: str  # email address or phone number
    type: str    # "email" or "phone"

class OTPVerifyRequest(BaseModel):
    target: str
    code: str
    name: Optional[str] = None

class GoogleAuthRequest(BaseModel):
    email: Optional[str] = None
    name: Optional[str] = None
    credential: Optional[str] = None  # Google OAuth ID Token
    avatar_url: Optional[str] = None

def send_real_email_otp(email: str, code: str):
    """Delivers Email OTP using Resend API if API key is provided."""
    if not settings.RESEND_API_KEY:
        return False
    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "from": "DocuAI Auth <auth@docuai.app>",
                "to": [email],
                "subject": f"{code} is your DocuAI Verification Code",
                "html": f"""
                <div style="font-family: sans-serif; padding: 20px;">
                    <h2>DocuAI Verification Code</h2>
                    <p>Your one-time security code is:</p>
                    <h1 style="font-size: 32px; color: #4F46E5; letter-spacing: 4px;">{code}</h1>
                    <p>This code expires in 10 minutes.</p>
                </div>
                """
            },
            timeout=5
        )
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"[AUTH ERROR] Failed to deliver Email OTP via Resend: {e}")
        return False

def send_real_sms_otp(phone: str, code: str):
    """Delivers SMS OTP using Twilio API if SID and token are provided."""
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.TWILIO_PHONE_NUMBER]):
        return False
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json"
        response = requests.post(
            url,
            auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            data={
                "From": settings.TWILIO_PHONE_NUMBER,
                "To": phone,
                "Body": f"Your DocuAI verification code is {code}. Valid for 10 minutes."
            },
            timeout=5
        )
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"[AUTH ERROR] Failed to deliver SMS OTP via Twilio: {e}")
        return False

@router.post("/otp/send")
def send_otp(request: OTPSendRequest):
    target = request.target.strip().lower()
    if not target:
        raise HTTPException(status_code=400, detail="Email or phone number is required.")
        
    code = f"{random.randint(100000, 999999)}"
    expires_at = time.time() + 600  # 10 mins validity
    
    otp_store[target] = {
        "code": code,
        "expires_at": expires_at
    }
    
    sent_via_provider = False
    if request.type == "email":
        sent_via_provider = send_real_email_otp(target, code)
    elif request.type == "phone":
        sent_via_provider = send_real_sms_otp(target, code)
        
    print(f"[AUTH LOG] Verification code for {target} ({request.type}): {code} (Provider Sent: {sent_via_provider})")
    
    return {
        "message": f"OTP sent successfully to {target}",
        "target": target,
        "type": request.type,
        "sent_via_provider": sent_via_provider,
        "dev_code": code  # Fallback for dev environment testing
    }

@router.post("/otp/verify")
def verify_otp(request: OTPVerifyRequest):
    target = request.target.strip().lower()
    code = request.code.strip()
    
    record = otp_store.get(target)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP requested for this address/number or it has expired.")
        
    if time.time() > record["expires_at"]:
        del otp_store[target]
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new code.")
        
    if record["code"] != code and code != "123456":  # 123456 dev fallback
        raise HTTPException(status_code=400, detail="Invalid OTP code. Please try again.")
        
    if target in otp_store:
        del otp_store[target]
    
    user_id = f"usr_{hashlib.md5(target.encode()).hexdigest()[:12]}"
    user_name = request.name or (target.split('@')[0] if '@' in target else f"User {target[-4:]}")
    
    user = {
        "id": user_id,
        "target": target,
        "name": user_name,
        "auth_type": "email" if "@" in target else "phone",
        "created_at": time.time()
    }
    users_db[user_id] = user
    
    return {
        "message": "Authentication successful",
        "user": user,
        "token": f"token_{user_id}_{int(time.time())}"
    }

@router.post("/google")
def google_auth(request: GoogleAuthRequest):
    email = request.email
    name = request.name
    avatar_url = request.avatar_url

    # If a Google ID Token (credential) is passed from GIS/OAuth Client, verify it with Google APIs
    if request.credential:
        try:
            token_res = requests.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={request.credential}",
                timeout=5
            )
            if token_res.status_code == 200:
                google_info = token_res.json()
                email = google_info.get("email")
                name = google_info.get("name", email.split("@")[0] if email else "Google User")
                avatar_url = google_info.get("picture", avatar_url)
        except Exception as e:
            print(f"[AUTH ERROR] Failed to verify Google ID Token: {e}")

    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Google authentication requires a valid email.")
        
    user_id = f"usr_g_{hashlib.md5(email.encode()).hexdigest()[:12]}"
    user_name = name or email.split('@')[0]
    
    user = {
        "id": user_id,
        "target": email,
        "name": user_name,
        "auth_type": "google",
        "avatar_url": avatar_url,
        "created_at": time.time()
    }
    users_db[user_id] = user
    
    return {
        "message": "Google authentication successful",
        "user": user,
        "token": f"token_{user_id}_{int(time.time())}"
    }

@router.get("/me")
def get_me(x_session_id: str = Header(default="default_session")):
    user = users_db.get(x_session_id)
    if user:
        return {"user": user}
    return {
        "user": {
            "id": x_session_id,
            "name": "Guest User",
            "auth_type": "guest"
        }
    }
