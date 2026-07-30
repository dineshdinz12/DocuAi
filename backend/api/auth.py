from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import random
import time
import hashlib
import uuid

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# In-memory OTP storage for dev & lightweight production (target -> {code, expires_at})
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
    email: str
    name: Optional[str] = None
    google_id: Optional[str] = None
    avatar_url: Optional[str] = None

@router.post("/otp/send")
def send_otp(request: OTPSendRequest):
    target = request.target.strip().lower()
    if not target:
        raise HTTPException(status_code=400, detail="Email or phone number is required.")
        
    # Generate 6-digit OTP
    code = f"{random.randint(100000, 999999)}"
    expires_at = time.time() + 600  # 10 minutes validity
    
    otp_store[target] = {
        "code": code,
        "expires_at": expires_at
    }
    
    # Print to backend log for instant dev testing
    print(f"[AUTH OTP] Verification code for {target} ({request.type}): {code}")
    
    return {
        "message": f"OTP sent successfully to {target}",
        "target": target,
        "type": request.type,
        "dev_code": code  # Included for immediate UI testing
    }

@router.post("/otp/verify")
def verify_otp(request: OTPVerifyRequest):
    target = request.target.strip().lower()
    code = request.code.strip()
    
    record = otp_store.get(target)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP requested for this address/number or it expired.")
        
    if time.time() > record["expires_at"]:
        del otp_store[target]
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new code.")
        
    if record["code"] != code and code != "123456":  # 123456 as universal dev fallback
        raise HTTPException(status_code=400, detail="Invalid OTP code. Please try again.")
        
    # Remove used OTP
    del otp_store[target]
    
    # Create or fetch persistent user
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
    email = request.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Google authentication requires a valid email.")
        
    user_id = f"usr_g_{hashlib.md5(email.encode()).hexdigest()[:12]}"
    user_name = request.name or email.split('@')[0]
    
    user = {
        "id": user_id,
        "target": email,
        "name": user_name,
        "auth_type": "google",
        "avatar_url": request.avatar_url,
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
