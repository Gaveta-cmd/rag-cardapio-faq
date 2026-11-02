"""API do assistente: POST /perguntar busca os trechos, monta o prompt e o LLM responde citando a fonte.

Variáveis: DATABASE_URL, ANTHROPIC_API_KEY, BUSCA_MODO (vetor ou hibrida) e LIMITE_SIMILARIDADE.
Uso: uvicorn rag.api:app --port 8000
"""
import os
from functools import lru_cache
from typing import Callable

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from rag import resposta as rsp
from rag.busca import Buscador, Resultado

TRECHOS_POR_PERGUNTA = 5
LIMITE_SIMILARIDADE = float(os.environ.get("LIMITE_SIMILARIDADE", "0.80"))

app = FastAPI(title="Assistente de cardápio e FAQ")


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500)


class Fonte(BaseModel):
    id: str
    titulo: str
    trecho: str


class Saida(BaseModel):
    resposta: str
    fontes: list[Fonte]
    sabe: bool
    motivo: str


@lru_cache
def obter_buscador() -> Buscador:
    from rag.banco import buscar_por_vetor, conectar, listar_trechos
    from rag.embeddings import Embeddings

    con = conectar()
    trechos = listar_trechos(con)
    if not trechos:
        raise RuntimeError("base vazia: rode python -m rag.indexar")
    embeddings = Embeddings()
    return Buscador(trechos, embeddings.de_pergunta, lambda v, n: buscar_por_vetor(con, v, n), os.environ.get("BUSCA_MODO", "hibrida"))


def obter_llm() -> Callable[[str, str], str]:
    return rsp.chamar_claude


def responder_pergunta(pergunta: str, resultado: Resultado, chamar_llm: Callable[[str, str], str], limite: float = LIMITE_SIMILARIDADE) -> rsp.Resposta:
    # Sem trecho relevante, nem chama o LLM: não gasta e não dá chance de inventar
    if resultado.similaridade_maxima < limite or not resultado.trechos:
        return rsp.nao_sei("sem_trecho_relevante")
    return rsp.responder(pergunta, resultado.trechos, chamar_llm)


@app.get("/saude")
def saude() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/perguntar", response_model=Saida)
def perguntar(entrada: Pergunta, buscador: Buscador = Depends(obter_buscador), chamar_llm: Callable[[str, str], str] = Depends(obter_llm)) -> Saida:
    resultado = buscador.buscar(entrada.pergunta, TRECHOS_POR_PERGUNTA)
    r = responder_pergunta(entrada.pergunta, resultado, chamar_llm)
    return Saida(resposta=r.texto, fontes=[Fonte(id=f.id, titulo=f.titulo, trecho=f.texto) for f in r.fontes], sabe=r.sabe, motivo=r.motivo)
