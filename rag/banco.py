"""Acesso ao Postgres com pgvector. Roda igual no Supabase e no Docker: só muda a DATABASE_URL."""
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from rag.documentos import Trecho

ESQUEMA = Path(__file__).resolve().parent.parent / "supabase" / "migrations" / "0001_trechos.sql"


def conectar() -> psycopg.Connection:
    con = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    register_vector(con)
    return con


def aplicar_esquema(con: psycopg.Connection) -> None:
    con.execute(ESQUEMA.read_text(encoding="utf-8"))
    register_vector(con)


def salvar_trechos(con: psycopg.Connection, trechos: list[Trecho], vetores: list[list[float]]) -> None:
    """Substitui tudo em uma transação: se algo falhar, a base antiga continua no ar."""
    with con.transaction(), con.cursor() as cur:
        cur.execute("delete from trechos")
        cur.executemany("insert into trechos (id, titulo, texto, embedding) values (%s, %s, %s, %s::vector)",
                        [(t.id, t.titulo, t.texto, str(v)) for t, v in zip(trechos, vetores)])


def listar_trechos(con: psycopg.Connection) -> list[Trecho]:
    return [Trecho(*linha) for linha in con.execute("select id, titulo, texto from trechos order by id").fetchall()]


def buscar_por_vetor(con: psycopg.Connection, vetor: list[float], limite: int) -> list[tuple[str, float]]:
    """Devolve (id, similaridade do cosseno), da maior para a menor."""
    return [tuple(linha) for linha in con.execute("select id, similaridade from buscar_trechos(%s::vector, %s)", (str(vetor), limite)).fetchall()]
