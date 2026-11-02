"""Aplica o esquema, gera os embeddings e grava os trechos no banco.

Uso: python -m rag.indexar
"""
from rag.banco import aplicar_esquema, conectar, listar_trechos, salvar_trechos
from rag.documentos import carregar_trechos
from rag.embeddings import DIMENSAO, Embeddings


def main() -> None:
    trechos = carregar_trechos()
    vetores = Embeddings().de_trechos([t.texto for t in trechos])
    if len(vetores) != len(trechos) or any(len(v) != DIMENSAO for v in vetores):
        raise ValueError("embeddings com tamanho inesperado")
    con = conectar()
    aplicar_esquema(con)
    salvar_trechos(con, trechos, vetores)
    gravados = len(listar_trechos(con))
    if gravados != len(trechos):
        raise ValueError(f"{gravados} trechos no banco, esperado {len(trechos)}")
    print(f"{gravados} trechos indexados")


if __name__ == "__main__":
    main()
