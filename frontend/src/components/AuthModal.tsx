"use client";

import { useState } from "react";
import { X, Lock, Mail, Loader2, CheckCircle2 } from "lucide-react";
import { setAuthenticatedUser, UserProfile } from "@/utils/session";
import { auth, googleProvider, signInWithPopup } from "@/utils/firebase";
import { supabase } from "@/lib/supabase";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (user: UserProfile) => void;
}

export function AuthModal({ isOpen, onClose, onSuccess }: AuthModalProps) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<"main" | "email" | "sent">("main");

  if (!isOpen) return null;

  // ── Google Sign-In via Firebase (primary — was already working) ──────────
  const handleGoogleLogin = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await signInWithPopup(auth, googleProvider);
      const fbUser = result.user;

      const profile: UserProfile = {
        id: `usr_fb_${fbUser.uid}`,
        name: fbUser.displayName || fbUser.email?.split("@")[0] || "Google User",
        target: fbUser.email || undefined,
        auth_type: "google",
        avatar_url: fbUser.photoURL || undefined,
      };

      setAuthenticatedUser(profile);
      onSuccess(profile);
      onClose();
    } catch (err: any) {
      console.warn("[Firebase] Google sign-in failed:", err.message);
      setError("Google sign-in failed. Make sure popups are allowed and try again.");
    } finally {
      setLoading(false);
    }
  };

  // ── Magic Link via Supabase (secondary / email fallback) ─────────────────
  const handleMagicLink = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const { error: sbError } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: {
          shouldCreateUser: true,
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      });

      if (sbError) throw sbError;
      setStep("sent");
    } catch (err: any) {
      setError(err.message || "Failed to send magic link. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-[400px] bg-white rounded-2xl shadow-2xl border border-slate-200/80 overflow-hidden transition-all">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" strokeWidth={1.75} />
        </button>

        {/* Modal Header */}
        <div className="px-8 pt-8 pb-5 text-center">
          <img 
            src="/logo-transparent.png" 
            alt="DocuAI Logo" 
            className="w-14 h-14 mx-auto mb-3 object-contain" 
          />
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">
            Welcome to DocuAI
          </h2>
          <p className="mt-1 text-xs text-slate-500 font-normal leading-relaxed">
            Sign in to access your document workspace and AI assistant
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-8 mb-4 p-3 bg-rose-50 border border-rose-200/60 text-rose-700 text-xs rounded-xl text-center font-medium">
            {error}
          </div>
        )}

        <div className="px-8 pb-8 space-y-3">

          {/* ── Main Step ──────────────────────────────────────────── */}
          {step === "main" && (
            <>
              {/* Google Sign-In Button (Primary) */}
              <button
                onClick={handleGoogleLogin}
                disabled={loading}
                className="w-full flex items-center justify-center gap-3 py-3 px-4 bg-white hover:bg-slate-50 border border-slate-200 hover:border-slate-300 rounded-xl font-medium text-sm text-slate-800 shadow-sm transition-all duration-150 active:scale-[0.99] disabled:opacity-50 cursor-pointer"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin text-slate-600" />
                ) : (
                  <svg className="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
                  </svg>
                )}
                <span>{loading ? "Connecting to Google..." : "Continue with Google"}</span>
              </button>

              <div className="flex items-center gap-3 py-0.5">
                <div className="flex-1 h-px bg-slate-200" />
                <span className="text-[11px] text-slate-400 font-medium">or</span>
                <div className="flex-1 h-px bg-slate-200" />
              </div>

              {/* Email Magic Link Button */}
              <button
                onClick={() => setStep("email")}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2.5 py-2.5 px-4 bg-slate-900 hover:bg-slate-800 text-white rounded-xl font-medium text-sm transition-all duration-150 active:scale-[0.99] disabled:opacity-50 cursor-pointer"
              >
                <Mail className="w-4 h-4" strokeWidth={1.5} />
                <span>Sign in with Email Link</span>
              </button>
            </>
          )}

          {/* ── Email Input Step ─────────────────────────────────── */}
          {step === "email" && (
            <>
              <button
                onClick={() => { setStep("main"); setError(null); }}
                className="text-xs text-slate-500 hover:text-slate-800 transition-colors flex items-center gap-1 mb-1"
              >
                ← Back
              </button>
              <form onSubmit={handleMagicLink} className="space-y-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Email address
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" strokeWidth={1.5} />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com"
                      required
                      autoFocus
                      className="w-full pl-9 pr-3 py-2.5 border border-slate-200 rounded-xl text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-slate-400 focus:ring-2 focus:ring-slate-100 transition-all"
                    />
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={loading || !email.trim()}
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-900 hover:bg-slate-800 text-white rounded-xl font-medium text-sm transition-all disabled:opacity-50 cursor-pointer"
                >
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Mail className="w-4 h-4" strokeWidth={1.5} />}
                  <span>{loading ? "Sending..." : "Send Magic Link"}</span>
                </button>
              </form>
            </>
          )}

          {/* ── Sent Confirmation ────────────────────────────────── */}
          {step === "sent" && (
            <div className="text-center py-4 space-y-3">
              <div className="w-12 h-12 bg-emerald-50 rounded-full flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-6 h-6 text-emerald-600" strokeWidth={1.75} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900">Check your email</p>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  We sent a magic link to <span className="font-semibold text-slate-700">{email}</span>. Click it to sign in automatically.
                </p>
              </div>
              <button
                onClick={() => { setStep("email"); setError(null); }}
                className="text-xs text-slate-500 hover:text-slate-800 transition-colors underline underline-offset-2"
              >
                Use a different email
              </button>
            </div>
          )}

          {/* Security Note */}
          <div className="pt-1 flex items-center justify-center gap-1.5 text-[11px] text-slate-400 font-normal">
            <Lock className="w-3 h-3 text-slate-400" strokeWidth={1.5} />
            <span>Secure, encrypted enterprise workspace</span>
          </div>
        </div>

        {/* Footer */}
        <div className="px-8 py-3 bg-slate-50 border-t border-slate-100 text-center text-[10px] text-slate-400">
          By continuing, you agree to DocuAI's Terms of Service and Privacy Policy.
        </div>
      </div>
    </div>
  );
}
