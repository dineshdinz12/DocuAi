"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { setAuthenticatedUser } from "@/utils/session";

export default function AuthCallbackPage() {
  const router = useRouter();

  useEffect(() => {
    const handleCallback = async () => {
      const { data: { session }, error } = await supabase.auth.getSession();

      if (session?.user) {
        const u = session.user;
        setAuthenticatedUser({
          id: u.id,
          name: u.user_metadata?.full_name || u.user_metadata?.name || u.email?.split("@")[0] || "User",
          target: u.email || undefined,
          auth_type: "google",
          avatar_url: u.user_metadata?.avatar_url || undefined,
        });
      }

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
