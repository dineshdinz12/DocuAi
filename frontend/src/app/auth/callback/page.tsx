"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function AuthCallbackPage() {
  const router = useRouter();
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const handleCallback = async () => {
      const url = new URL(window.location.href);
      const code = url.searchParams.get("code");
      const result = code
        ? await supabase.auth.exchangeCodeForSession(code)
        : await supabase.auth.getSession();
      if (result.error || !result.data.session) {
        setFailed(true);
        return;
      }

      router.replace("/");
    };

    void handleCallback().catch(() => setFailed(true));
  }, [router]);

  return (
    <div className="flex items-center justify-center h-screen bg-white">
      <div className="text-center space-y-3">
        {failed ? (
          <>
            <p role="alert" className="text-sm text-slate-700">Google sign-in could not be completed.</p>
            <button onClick={() => router.replace("/")} className="rounded-lg bg-slate-900 px-4 py-2 text-sm text-white">Try again</button>
          </>
        ) : (
          <>
            <div className="w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full animate-spin mx-auto" />
            <p role="status" className="text-sm text-slate-500 font-medium">Signing you in to DocuAI…</p>
          </>
        )}
      </div>
    </div>
  );
}
