"use client";

import { useState, useEffect } from "react";
import { X, Mail, Phone, Lock, Sparkles, CheckCircle2, ArrowRight } from "lucide-react";
import { setAuthenticatedUser, UserProfile } from "@/utils/session";
import { auth, googleProvider, signInWithPopup, RecaptchaVerifier, signInWithPhoneNumber, ConfirmationResult } from "@/utils/firebase";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (user: UserProfile) => void;
}

declare global {
  interface Window {
    recaptchaVerifier?: RecaptchaVerifier;
  }
}

export function AuthModal({ isOpen, onClose, onSuccess }: AuthModalProps) {
  const [tab, setTab] = useState<"google" | "email" | "phone">("google");
  const [target, setTarget] = useState("");
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [step, setStep] = useState<"input" | "otp">("input");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devCode, setDevCode] = useState<string | null>(null);
  const [confirmationResult, setConfirmationResult] = useState<ConfirmationResult | null>(null);

  if (!isOpen) return null;

  const resetState = () => {
    setStep("input");
    setTarget("");
    setCode("");
    setError(null);
    setDevCode(null);
    setConfirmationResult(null);
  };

  const parseJsonResponse = async (res: Response) => {
    const contentType = res.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      return await res.json();
    }
    throw new Error(`Server returned HTTP ${res.status}. Please ensure backend is running.`);
  };

  // Google Authentication via Firebase
  const handleGoogleLogin = async () => {
    setLoading(true);
    setError(null);

    try {
      // Attempt Firebase Google OAuth Popup
      const result = await signInWithPopup(auth, googleProvider);
      const fbUser = result.user;

      const profile: UserProfile = {
        id: `usr_fb_${fbUser.uid}`,
        name: fbUser.displayName || fbUser.email?.split("@")[0] || "Google User",
        target: fbUser.email || undefined,
        auth_type: "google",
        avatar_url: fbUser.photoURL || undefined
      };

      setAuthenticatedUser(profile);
      onSuccess(profile);
      onClose();
      resetState();
    } catch (err: any) {
      console.warn("[Firebase Auth] Falling back to backend Google auth:", err.message);
      
      // Fallback for dev environment without API credentials
      try {
        const mockEmail = target || `user.google.${Math.floor(Math.random() * 1000)}@gmail.com`;
        const res = await fetch("/api/v1/auth/google", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: mockEmail, name: "Google Account User" }),
        });

        const data = await parseJsonResponse(res);
        if (!res.ok) throw new Error(data.detail || "Google authentication failed");

        setAuthenticatedUser(data.user);
        onSuccess(data.user);
        onClose();
        resetState();
      } catch (backendErr: any) {
        setError(backendErr.message || "Google authentication failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Send Phone SMS OTP via Firebase or Backend
  const handleSendOTP = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!target.trim()) {
      setError(`Please enter a valid ${tab === "email" ? "email address" : "phone number"}`);
      return;
    }

    setLoading(true);
    setError(null);

    if (tab === "phone") {
      try {
        if (!window.recaptchaVerifier) {
          window.recaptchaVerifier = new RecaptchaVerifier(auth, "recaptcha-container", {
            size: "invisible",
          });
        }
        const confirmation = await signInWithPhoneNumber(auth, target, window.recaptchaVerifier);
        setConfirmationResult(confirmation);
        setStep("otp");
        setLoading(false);
        return;
      } catch (fbPhoneErr: any) {
        console.warn("[Firebase Phone Auth] Falling back to backend SMS OTP service:", fbPhoneErr.message);
      }
    }

    // Backend OTP Request (Email or Phone fallback)
    try {
      const res = await fetch("/api/v1/auth/otp/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target, type: tab }),
      });

      const data = await parseJsonResponse(res);
      if (!res.ok) throw new Error(data.detail || "Failed to send OTP code");

      setDevCode(data.dev_code || "123456");
      setStep("otp");
    } catch (err: any) {
      setError(err.message || "Failed to send verification code");
    } finally {
      setLoading(false);
    }
  };

  // Verify Phone/Email OTP
  const handleVerifyOTP = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim()) {
      setError("Please enter the 6-digit code");
      return;
    }

    setLoading(true);
    setError(null);

    // If Firebase Phone Auth confirmation exists
    if (confirmationResult) {
      try {
        const result = await confirmationResult.confirm(code);
        const fbUser = result.user;

        const profile: UserProfile = {
          id: `usr_fb_${fbUser.uid}`,
          name: name || fbUser.phoneNumber || "Firebase Phone User",
          target: fbUser.phoneNumber || target,
          auth_type: "phone"
        };

        setAuthenticatedUser(profile);
        onSuccess(profile);
        onClose();
        resetState();
        return;
      } catch (fbVerifyErr: any) {
        console.warn("[Firebase OTP Verify] Firebase verification failed, testing backend verification:", fbVerifyErr.message);
      }
    }

    // Backend OTP Verification
    try {
      const res = await fetch("/api/v1/auth/otp/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target, code, name }),
      });

      const data = await parseJsonResponse(res);
      if (!res.ok) throw new Error(data.detail || "Verification failed");

      setAuthenticatedUser(data.user);
      onSuccess(data.user);
      onClose();
      resetState();
    } catch (err: any) {
      setError(err.message || "Invalid code");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div id="recaptcha-container"></div>
      <div className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-200">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900 leading-tight">Sign In to DocuAI</h2>
              <p className="text-xs text-gray-500">Firebase Auth · Google, Email & Free SMS OTP</p>
            </div>
          </div>
          <button
            onClick={() => {
              onClose();
              resetState();
            }}
            className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Auth Method Navigation Tabs */}
        <div className="flex border-b border-gray-100 bg-gray-50/50 p-1">
          <button
            onClick={() => { setTab("google"); resetState(); }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              tab === "google" ? "bg-white text-indigo-600 shadow-sm" : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Google Auth
          </button>
          <button
            onClick={() => { setTab("email"); resetState(); }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              tab === "email" ? "bg-white text-indigo-600 shadow-sm" : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Email OTP
          </button>
          <button
            onClick={() => { setTab("phone"); resetState(); }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              tab === "phone" ? "bg-white text-indigo-600 shadow-sm" : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Phone OTP
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6">
          {error && (
            <div className="mb-4 p-3 text-xs bg-red-50 text-red-600 border border-red-100 rounded-xl">
              {error}
            </div>
          )}

          {/* TAB 1: GOOGLE AUTH */}
          {tab === "google" && (
            <div className="space-y-4 text-center py-2">
              <p className="text-xs text-gray-600 leading-relaxed">
                Sign in with your Google account via Firebase Auth to automatically sync your documents and chat memory.
              </p>
              
              <button
                onClick={handleGoogleLogin}
                disabled={loading}
                className="w-full flex items-center justify-center gap-3 py-3 px-4 bg-white border border-gray-300 rounded-xl hover:bg-gray-50 hover:border-gray-400 font-semibold text-sm text-gray-700 transition-all shadow-sm active:scale-[0.99] disabled:opacity-50"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
                </svg>
                {loading ? "Signing in with Firebase..." : "Continue with Google"}
              </button>
            </div>
          )}

          {/* TAB 2 & 3: EMAIL / PHONE OTP */}
          {(tab === "email" || tab === "phone") && (
            <div>
              {step === "input" ? (
                <form onSubmit={handleSendOTP} className="space-y-4">
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wider">
                      {tab === "email" ? "Email Address" : "Phone Number (with Country Code)"}
                    </label>
                    <div className="relative">
                      {tab === "email" ? (
                        <Mail className="w-4 h-4 text-gray-400 absolute left-3.5 top-3.5" />
                      ) : (
                        <Phone className="w-4 h-4 text-gray-400 absolute left-3.5 top-3.5" />
                      )}
                      <input
                        type={tab === "email" ? "email" : "tel"}
                        value={target}
                        onChange={(e) => setTarget(e.target.value)}
                        placeholder={tab === "email" ? "dinesh@example.com" : "+91 9876543210"}
                        className="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-gray-900 placeholder-gray-400"
                        required
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wider">
                      Your Name (Optional)
                    </label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Dinesh"
                      className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-gray-900 placeholder-gray-400"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded-xl transition-all shadow-md shadow-indigo-200 disabled:opacity-50"
                  >
                    {loading ? "Sending OTP..." : "Send Verification Code"}
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </form>
              ) : (
                <form onSubmit={handleVerifyOTP} className="space-y-4">
                  <div className="p-3 bg-indigo-50 border border-indigo-100 rounded-xl text-xs text-indigo-800 flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold">Code sent to {target}</p>
                      {devCode && <p className="mt-0.5 text-indigo-600 font-mono">Dev Test OTP: {devCode}</p>}
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wider">
                      Enter 6-Digit OTP Code
                    </label>
                    <div className="relative">
                      <Lock className="w-4 h-4 text-gray-400 absolute left-3.5 top-3.5" />
                      <input
                        type="text"
                        maxLength={6}
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                        placeholder="123456"
                        className="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm font-mono tracking-widest text-center focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-gray-900"
                        autoFocus
                        required
                      />
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setStep("input")}
                      className="w-1/3 py-2.5 px-3 border border-gray-200 rounded-xl text-xs font-semibold text-gray-600 hover:bg-gray-50 transition-colors"
                    >
                      Back
                    </button>
                    <button
                      type="submit"
                      disabled={loading}
                      className="w-2/3 py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded-xl transition-all shadow-md shadow-indigo-200 disabled:opacity-50"
                    >
                      {loading ? "Verifying..." : "Verify & Sign In"}
                    </button>
                  </div>
                </form>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
