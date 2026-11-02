"""Embeddings multilíngues. A família E5 pede os prefixos "passage: " e "query: "; sem eles, a busca piora."""
import os

MODELO = os.environ.get("EMBEDDINGS_MODELO", "intfloat/multilingual-e5-small")
DIMENSAO = 384


class Embeddings:
    def __init__(self, modelo: str = MODELO):
        from sentence_transformers import SentenceTransformer  # importa aqui: os testes não precisam do torch

        self.modelo = SentenceTransformer(modelo, device="cpu")

    def _codificar(self, textos: list[str]) -> list[list[float]]:
        return self.modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False).tolist()

    def de_trechos(self, textos: list[str]) -> list[list[float]]:
        return self._codificar([f"passage: {t}" for t in textos])

    def de_pergunta(self, pergunta: str) -> list[float]:
        return self._codificar([f"query: {pergunta}"])[0]
