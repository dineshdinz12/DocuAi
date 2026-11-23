import { supabase } from "./supabase";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

export async function authHeaders(extra: Record<string, string> = {}) {
  const { data } = await supabase.auth.getSession();
  if (!data.session?.access_token) throw new Error("Please sign in again.");
  return { ...extra, Authorization: `Bearer ${data.session.access_token}` };
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  const headers = await authHeaders(init.headers as Record<string, string> | undefined);
  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${response.status})`);
  }
  return response;
}
