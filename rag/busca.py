"""Busca dos trechos: só por vetor (versão inicial) ou híbrida, com BM25 e vetor (a melhoria medida na avaliação)."""
import re
from dataclasses import dataclass
from typing import Callable

from rank_bm25 import BM25Okapi

from rag.documentos import Trecho, normalizar

CONSTANTE_RRF = 60  # valor usual da fusão por posição (reciprocal rank fusion)
CANDIDATOS = 20


@dataclass(frozen=True)
class Resultado:
    trechos: list[Trecho]
    similaridade_maxima: float  # do vetor; decide se existe algo relevante, nos dois modos


def tokenizar(texto: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalizar(texto))


def fundir_por_posicao(*listas: list[str]) -> list[str]:
    """Reciprocal rank fusion: combina listas pela posição, sem precisar comparar escalas de nota diferentes."""
    notas: dict[str, float] = {}
    for lista in listas:
        for posicao, item in enumerate(lista):
            notas[item] = notas.get(item, 0.0) + 1 / (CONSTANTE_RRF + posicao + 1)
    return sorted(notas, key=lambda i: (-notas[i], i))


class Buscador:
    def __init__(self, trechos: list[Trecho], vetorizar: Callable[[str], list[float]],
                 buscar_vetor: Callable[[list[float], int], list[tuple[str, float]]], modo: str = "hibrida"):
        if modo not in ("vetor", "hibrida"):
            raise ValueError(f"modo desconhecido: {modo}")
        self.por_id = {t.id: t for t in trechos}
        self.ids = [t.id for t in trechos]
        self.vetorizar, self.buscar_vetor, self.modo = vetorizar, buscar_vetor, modo
        self.bm25 = BM25Okapi([tokenizar(t.texto) for t in trechos])

    def _por_palavras(self, pergunta: str, limite: int) -> list[str]:
        notas = self.bm25.get_scores(tokenizar(pergunta))
        ordem = sorted(range(len(self.ids)), key=lambda i: -notas[i])
        return [self.ids[i] for i in ordem[:limite] if notas[i] > 0]

    def buscar(self, pergunta: str, k: int) -> Resultado:
        pelo_vetor = self.buscar_vetor(self.vetorizar(pergunta), CANDIDATOS)
        ordem = [i for i, _ in pelo_vetor]
        if self.modo == "hibrida":
            ordem = fundir_por_posicao(ordem, self._por_palavras(pergunta, CANDIDATOS))
        maxima = max((s for _, s in pelo_vetor), default=0.0)
        return Resultado([self.por_id[i] for i in ordem[:k]], maxima)
