"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function AuthCallbackPage() {
  const router = useRouter();

  useEffect(() => {
    const handleCallback = async () => {
      const code = new URL(window.location.href).searchParams.get("code");
      if (code) await supabase.auth.exchangeCodeForSession(code);
      else await supabase.auth.getSession();

      router.replace("/");
    };

    handleCallback();
  }, [router]);

  return (
    <div className="flex items-center justify-center h-screen bg-white">
      <div className="text-center space-y-3">
        <div className="w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm text-slate-500 font-medium">Signing you in...</p>
      </div>
    </div>
  );
}
