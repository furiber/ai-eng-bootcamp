-- Schema for the study-plan agent: turns a parent's intake note into a saved plan and a
-- placement test for the first lesson.
--
-- Run this ONCE against the Supabase project named by SUPABASE_PROJECT_REF in .env, using
-- the SQL Editor in the dashboard. It is idempotent, and it extends the tables created by
-- capstone_schema.sql -- run that one first.
--
-- Conventions follow Supabase's Postgres guidance: lowercase snake_case identifiers,
-- bigint identity primary keys, timestamptz for time, indexes on every foreign key.

-- The learner table already exists (capstone_schema.sql). A kid needs two more facts, and
-- both stay nullable so existing rows are unaffected.
alter table public.learner add column if not exists age       int;
alter table public.learner add column if not exists interests text;

-- Age is a sanity check, not a business rule: this tutor is aimed at children, and an
-- out-of-range value almost always means the agent misread the intake.
alter table public.learner drop constraint if exists learner_age_check;
alter table public.learner add  constraint learner_age_check check (age is null or age between 3 and 18);

create table if not exists public.study_plan (
  id                  bigint generated always as identity primary key,
  learner_id          bigint not null references public.learner (id) on delete cascade,
  -- CEFR level the agent judged the kid is at, from age + intake + any review history.
  assessed_level      text not null check (assessed_level in ('A1', 'A2', 'B1', 'B2', 'C1')),
  sessions_per_week   int  not null check (sessions_per_week between 1 and 7),
  -- Kids' attention spans set this range; the assignment fixes it at 10-20 minutes.
  minutes_per_session int  not null check (minutes_per_session between 10 and 20),
  -- The availability the recommendation was derived from, in the parent's own words.
  availability_note   text not null,
  -- Why this cadence and this level. Written for the parent to read.
  rationale           text not null,
  created_at          timestamptz not null default now()
);
create index if not exists study_plan_learner_id_idx on public.study_plan (learner_id);

-- The first session: a short test that checks whether assessed_level was right.
create table if not exists public.placement_test (
  id           bigint generated always as identity primary key,
  plan_id      bigint not null references public.study_plan (id) on delete cascade,
  -- When the agent suggests running it, in the parent's own terms ("Monday after school").
  session_note text not null,
  created_at   timestamptz not null default now()
);
create index if not exists placement_test_plan_id_idx on public.placement_test (plan_id);

create table if not exists public.placement_item (
  id               bigint generated always as identity primary key,
  test_id          bigint not null references public.placement_test (id) on delete cascade,
  position         int  not null check (position between 1 and 20),
  spanish          text not null,
  -- Must come from the Wiktionary lookup tool, not from the model's memory.
  expected_english text not null,
  -- What the item is probing, e.g. "greeting", "animal noun", "present-tense verb".
  skill            text not null,
  unique (test_id, position)
);
create index if not exists placement_item_test_id_idx on public.placement_item (test_id);

-- RLS on, no policies. Nothing reaches these tables through the anon or authenticated
-- keys. The agent connects through the Supabase MCP server with a personal access token,
-- which is not subject to RLS. Add per-learner policies when the app grows a real login.
alter table public.study_plan     enable row level security;
alter table public.placement_test enable row level security;
alter table public.placement_item enable row level security;
