from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from services.supabase_service import get_supabase


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None = None


def require_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    token = authorization.removeprefix("Bearer ").strip()
    client = get_supabase()
    if client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service is unavailable")

    try:
        response = client.auth.get_user(token)
        user = response.user
        if not user:
            raise ValueError("Missing user")
        return CurrentUser(id=str(user.id), email=user.email)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token")


AuthenticatedUser = Depends(require_user)
