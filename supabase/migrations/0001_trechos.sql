-- Trechos dos documentos do restaurante e a busca por similaridade.
create extension if not exists vector;

create table if not exists trechos (
    id text primary key,
    titulo text not null,
    texto text not null,
    embedding vector(384) not null
);

-- A API acessa com a conexão do servidor. Sem política de RLS, a chave pública do Supabase não lê nem altera a tabela.
alter table trechos enable row level security;

create or replace function buscar_trechos(consulta vector(384), limite int)
returns table (id text, similaridade double precision)
language sql
stable
set search_path = public, extensions
as $$
    select t.id, 1 - (t.embedding <=> consulta)
    from trechos t
    order by t.embedding <=> consulta
    limit limite;
$$;
