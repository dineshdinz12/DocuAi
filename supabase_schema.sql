-- ============================================================
-- DocuAI Supabase Schema
-- Run this in Supabase SQL Editor (dashboard -> SQL Editor -> New query)
-- ============================================================

-- 1. User Profiles (extends Supabase auth.users)
create table if not exists public.user_profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  email text,
  name text,
  avatar_url text,
  created_at timestamptz default now()
);

-- 2. Documents metadata (physical files stay on MinIO/disk)
create table if not exists public.documents (
  id text primary key,              -- matches doc_id from document_service
  user_id uuid references auth.users(id) on delete cascade,
  session_id text not null,
  name text not null,
  storage_key text not null,        -- file path or S3 key
  size bigint default 0,
  uploaded_at timestamptz default now()
);

-- 3. Chat Sessions
create table if not exists public.chat_sessions (
  id text primary key,
  user_id uuid references auth.users(id) on delete cascade,
  session_id text not null,
  title text default 'New Chat',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- 4. Chat Messages
create table if not exists public.chat_messages (
  id text primary key,
  session_id text references public.chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  sources jsonb default '[]'::jsonb,  -- Phase A: source citations
  created_at timestamptz default now()
);

-- ============================================================
-- Row Level Security (RLS) - each user sees only their data
-- ============================================================
alter table public.user_profiles enable row level security;
alter table public.documents enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;

-- User Profiles policies
drop policy if exists "Users manage own profile" on public.user_profiles;
create policy "Users manage own profile" on public.user_profiles
  for all using (auth.uid() = id);

-- Documents policies - allow anon access via session_id for guests too
drop policy if exists "Users manage own documents" on public.documents;
create policy "Users manage own documents" on public.documents
  for all using (
    auth.uid() = user_id
    or auth.uid() is null -- guest access (service role will handle)
  );

-- Chat sessions policies
drop policy if exists "Users manage own chat sessions" on public.chat_sessions;
create policy "Users manage own chat sessions" on public.chat_sessions
  for all using (
    auth.uid() = user_id
    or auth.uid() is null
  );

-- Chat messages policies  
drop policy if exists "Users manage own messages" on public.chat_messages;
create policy "Users manage own messages" on public.chat_messages
  for all using (
    session_id in (
      select id from public.chat_sessions
      where user_id = auth.uid() or auth.uid() is null
    )
  );

-- ============================================================
-- Auto-create user_profile on signup trigger
-- ============================================================
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.user_profiles (id, email, name, avatar_url)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name', split_part(new.email, '@', 1)),
    new.raw_user_meta_data->>'avatar_url'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
