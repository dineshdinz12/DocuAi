export function getSessionId(): string {
  if (typeof window === "undefined") return "default_session";
  let sessionId = localStorage.getItem("docuai_session_id");
  if (!sessionId) {
    sessionId = `session_${Math.random().toString(36).substring(2, 11)}_${Date.now()}`;
    localStorage.setItem("docuai_session_id", sessionId);
  }
  return sessionId;
}
