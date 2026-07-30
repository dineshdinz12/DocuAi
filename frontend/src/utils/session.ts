export interface UserProfile {
  id: string;
  name: string;
  target?: string;
  auth_type: "google" | "email" | "phone" | "guest";
  avatar_url?: string;
}

export function getCurrentUser(): UserProfile | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem("docuai_user");
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

export function setAuthenticatedUser(user: UserProfile): void {
  if (typeof window === "undefined") return;
  localStorage.setItem("docuai_user", JSON.stringify(user));
  localStorage.setItem("docuai_session_id", user.id);
}

export function clearUserSession(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem("docuai_user");
  localStorage.removeItem("docuai_session_id");
}

export function getSessionId(): string {
  if (typeof window === "undefined") return "default_session";
  const user = getCurrentUser();
  if (user && user.id) {
    return user.id;
  }
  let sessionId = localStorage.getItem("docuai_session_id");
  if (!sessionId) {
    sessionId = `guest_${Math.random().toString(36).substring(2, 11)}_${Date.now()}`;
    localStorage.setItem("docuai_session_id", sessionId);
  }
  return sessionId;
}
