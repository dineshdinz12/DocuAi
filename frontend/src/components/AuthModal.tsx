"use client";

import { useState } from "react";
import Image from "next/image";
import { supabase } from "@/lib/supabase";

export function AuthModal({ isOpen }: { isOpen: boolean; onClose: () => void }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  if (!isOpen) return null;

  const signIn = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: `${window.location.origin}/auth/callback` },
      });
      if (result.error) throw result.error;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Google sign-in failed.");
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/25 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="signin-title">
      <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-xl">
        <Image src="/logo-transparent.png" alt="DocuAI" width={140} height={62} className="mx-auto h-14 w-auto object-contain" />
        <h2 id="signin-title" className="mt-5 text-xl font-semibold text-slate-950">Sign in to DocuAI</h2>
        <p className="mt-2 text-sm text-slate-600">Open your documents and conversations.</p>
        {error && <p role="alert" className="mt-4 text-sm text-rose-700">{error}</p>}
        <button onClick={() => void signIn()} disabled={loading} className="mt-6 w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60">
          {loading ? "Opening Google…" : "Continue with Google"}
        </button>
      </div>
    </div>
  );
}
