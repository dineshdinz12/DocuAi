"use client";

import { FormEvent, useState } from "react";
import { CheckCircle2, Loader2, Mail, ShieldCheck, X } from "lucide-react";
import { supabase } from "@/lib/supabase";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AuthModal({ isOpen, onClose }: AuthModalProps) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState<"google" | "email" | null>(null);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  if (!isOpen) return null;

  const signInWithGoogle = async () => {
    setLoading("google");
    setError("");
    const { error: authError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
    if (authError) {
      setError(authError.message);
      setLoading(null);
    }
  };

  const sendMagicLink = async (event: FormEvent) => {
    event.preventDefault();
    setLoading("email");
    setError("");
    const { error: authError } = await supabase.auth.signInWithOtp({
      email: email.trim(),
      options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
    });
    setLoading(null);
    if (authError) setError(authError.message);
    else setSent(true);
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/55 p-4 backdrop-blur-md" role="dialog" aria-modal="true" aria-label="Sign in">
      <div className="relative w-full max-w-md overflow-hidden rounded-[28px] border border-white/10 bg-[#101827] p-7 text-white shadow-2xl shadow-slate-950/40">
        <div className="absolute -right-16 -top-20 h-48 w-48 rounded-full bg-violet-500/20 blur-3xl" />
        <button onClick={onClose} className="absolute right-5 top-5 rounded-full p-2 text-slate-400 hover:bg-white/10 hover:text-white" aria-label="Close sign in">
          <X className="h-4 w-4" />
        </button>
        <div className="relative">
          <div className="mb-6 grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-500 shadow-lg shadow-violet-500/20">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <h2 className="text-2xl font-semibold tracking-tight">Your private document workspace</h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">Sign in to keep documents, conversations, and citations securely connected to your account.</p>
          {error && <p className="mt-4 rounded-xl border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">{error}</p>}
          {sent ? (
            <div className="mt-7 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-5 text-center">
              <CheckCircle2 className="mx-auto h-7 w-7 text-emerald-300" />
              <p className="mt-3 font-medium">Check your inbox</p>
              <p className="mt-1 text-sm text-slate-400">We sent a secure sign-in link to {email}.</p>
            </div>
          ) : (
            <div className="mt-7 space-y-4">
              <button onClick={signInWithGoogle} disabled={!!loading} className="flex h-12 w-full items-center justify-center gap-3 rounded-xl bg-white font-medium text-slate-900 transition hover:bg-slate-100 disabled:opacity-60">
                {loading === "google" ? <Loader2 className="h-4 w-4 animate-spin" /> : <span className="text-lg font-bold text-blue-500">G</span>}
                Continue with Google
              </button>
              <div className="flex items-center gap-3 text-xs text-slate-500"><span className="h-px flex-1 bg-white/10" />or use email<span className="h-px flex-1 bg-white/10" /></div>
              <form onSubmit={sendMagicLink} className="space-y-3">
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                  <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" required placeholder="you@company.com" className="h-12 w-full rounded-xl border border-white/10 bg-white/5 pl-11 pr-4 text-sm outline-none transition placeholder:text-slate-600 focus:border-violet-400/60 focus:ring-4 focus:ring-violet-500/10" />
                </div>
                <button disabled={!!loading} className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-violet-500 font-medium transition hover:bg-violet-400 disabled:opacity-60">
                  {loading === "email" && <Loader2 className="h-4 w-4 animate-spin" />} Send magic link
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
