-- Capstone schema: Spanish flashcard tutor (minimal slice for the deck-builder agent).
--
-- Run this ONCE against the Supabase project named by SUPABASE_PROJECT_REF in .env,
-- using the SQL Editor in the Supabase dashboard. It is idempotent.
--
-- Conventions follow Supabase's Postgres guidance: lowercase snake_case identifiers,
-- bigint identity primary keys, timestamptz for time, indexes on every foreign key.

create table if not exists public.learner (
  id         bigint generated always as identity primary key,
  email      text not null unique,
  name       text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.deck (
  id         bigint generated always as identity primary key,
  learner_id bigint not null references public.learner (id) on delete cascade,
  title      text not null,
  -- CEFR level. text + check constraint rather than a Postgres enum, so adding a
  -- level later is a one-line migration instead of an ALTER TYPE.
  level      text not null check (level in ('A1', 'A2', 'B1', 'B2', 'C1')),
  created_at timestamptz not null default now()
);
create index if not exists deck_learner_id_idx on public.deck (learner_id);

create table if not exists public.card (
  id         bigint generated always as identity primary key,
  deck_id    bigint not null references public.deck (id) on delete cascade,
  spanish    text not null,
  english    text not null,
  example_es text,
  unique (deck_id, spanish)
);
create index if not exists card_deck_id_idx on public.card (deck_id);

create table if not exists public.review (
  id          bigint generated always as identity primary key,
  learner_id  bigint not null references public.learner (id) on delete cascade,
  card_id     bigint not null references public.card (id) on delete cascade,
  correct     boolean not null,
  reviewed_at timestamptz not null default now()
);
create index if not exists review_learner_id_idx on public.review (learner_id);
create index if not exists review_card_id_idx    on public.review (card_id);

-- RLS on, no policies. Nothing reaches these tables through the anon or authenticated
-- keys. The agent connects through the Supabase MCP server with a personal access
-- token, which is not subject to RLS. Add per-learner policies when the app grows a
-- real login (auth.uid() = learner.auth_user_id).
alter table public.learner enable row level security;
alter table public.deck    enable row level security;
alter table public.card    enable row level security;
alter table public.review  enable row level security;

-- --- Seed: one learner with enough review history for the agent to judge a level ---

insert into public.learner (email, name)
values ('maria@example.com', 'Maria Lopez')
on conflict (email) do nothing;

insert into public.deck (learner_id, title, level)
select l.id, 'Everyday Nouns', 'A1'
from public.learner l
where l.email = 'maria@example.com'
  and not exists (
    select 1 from public.deck d
    where d.learner_id = l.id and d.title = 'Everyday Nouns'
  );

insert into public.card (deck_id, spanish, english, example_es)
select d.id, v.spanish, v.english, v.example_es
from public.deck d
join public.learner l on l.id = d.learner_id
cross join (values
  ('la casa',    'the house',  'La casa es grande.'),
  ('el perro',   'the dog',    'El perro corre en el parque.'),
  ('el agua',    'the water',  'Quiero un vaso de agua.'),
  ('la comida',  'the food',   'La comida esta lista.'),
  ('el libro',   'the book',   'Leo un libro cada noche.'),
  ('la ciudad',  'the city',   'La ciudad es muy ruidosa.')
) as v (spanish, english, example_es)
where l.email = 'maria@example.com' and d.title = 'Everyday Nouns'
on conflict (deck_id, spanish) do nothing;

-- 5 of 6 correct -> ~83% accuracy on A1, so the agent should promote her to A2.
insert into public.review (learner_id, card_id, correct)
select l.id, c.id, (c.spanish <> 'la ciudad')
from public.learner l
join public.deck d on d.learner_id = l.id and d.title = 'Everyday Nouns'
join public.card c on c.deck_id = d.id
where l.email = 'maria@example.com'
  and not exists (select 1 from public.review r where r.card_id = c.id and r.learner_id = l.id);
