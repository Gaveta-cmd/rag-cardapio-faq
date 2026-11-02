"""Avalia a busca e as respostas. O limite de similaridade é escolhido só com o conjunto "dev".

Uso: python -m avaliacao.avaliar busca --conjunto dev
     python -m avaliacao.avaliar respostas --conjunto teste --modo hibrida --limite 0.8
"""
import argparse
import json
from pathlib import Path

from rag.api import TRECHOS_POR_PERGUNTA, responder_pergunta
from rag.banco import buscar_por_vetor, conectar, listar_trechos
from rag.busca import Buscador
from rag.documentos import normalizar
from rag.embeddings import Embeddings
from rag.resposta import chamar_claude

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "outputs" / "resultados"
LIMITES = [0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]


def ler_perguntas(conjunto: str) -> list[dict]:
    linhas = (RAIZ / "avaliacao" / "perguntas.jsonl").read_text(encoding="utf-8").splitlines()
    return [p for p in map(json.loads, linhas) if p["conjunto"] == conjunto]


def posicao_da_fonte(perguntada: dict, ids: list[str]) -> int | None:
    return next((i for i, t in enumerate(ids) if t in perguntada["fontes"]), None)


def contem_os_fatos(resposta: str, deve_conter: list[list[str]]) -> bool:
    """Cada lista interna é um fato; basta uma das formas de escrever aquele fato."""
    texto = normalizar(resposta)
    return all(any(normalizar(forma) in texto for forma in formas) for formas in deve_conter)


def avaliar_busca(perguntas: list[dict], buscador: Buscador) -> dict:
    respondiveis = [p for p in perguntas if not p.get("sem_resposta")]
    acertos, inverso, detalhes = 0, 0.0, []
    similaridades = {}
    for p in perguntas:
        resultado = buscador.buscar(p["pergunta"], TRECHOS_POR_PERGUNTA)
        similaridades[p["id"]] = resultado.similaridade_maxima
        if p.get("sem_resposta"):
            continue
        posicao = posicao_da_fonte(p, [t.id for t in resultado.trechos])
        acertos += posicao is not None
        inverso += 0 if posicao is None else 1 / (posicao + 1)
        if posicao is None:
            detalhes.append({"id": p["id"], "pergunta": p["pergunta"], "fontes": p["fontes"], "trouxe": [t.id for t in resultado.trechos]})
    return {"n": len(respondiveis), "acerto_top_k": round(acertos / len(respondiveis), 4), "mrr": round(inverso / len(respondiveis), 4), "erros": detalhes, "similaridades": similaridades}


def varrer_limites(perguntas: list[dict], similaridades: dict) -> list[dict]:
    """Passa com o limite: as respondíveis deveriam passar, e as sem resposta deveriam barrar."""
    tabela = []
    for limite in LIMITES:
        passam = sum(similaridades[p["id"]] >= limite for p in perguntas if not p.get("sem_resposta"))
        barradas = sum(similaridades[p["id"]] < limite for p in perguntas if p.get("sem_resposta"))
        tabela.append({"limite": limite, "respondiveis_que_passam": passam, "sem_resposta_barradas": barradas})
    return tabela


def avaliar_respostas(perguntas: list[dict], buscador: Buscador, limite: float) -> dict:
    linhas = []
    for p in perguntas:
        resultado = buscador.buscar(p["pergunta"], TRECHOS_POR_PERGUNTA)
        r = responder_pergunta(p["pergunta"], resultado, chamar_claude, limite)
        if p.get("sem_resposta"):
            correta = not r.sabe
        else:
            correta = r.sabe and contem_os_fatos(r.texto, p["deve_conter"])
        linhas.append({"id": p["id"], "pergunta": p["pergunta"], "sem_resposta": bool(p.get("sem_resposta")), "sabe": r.sabe, "motivo": r.motivo, "correta": correta,
                       "resposta": r.texto, "contextos": [t.texto for t in resultado.trechos], "ids_contexto": [t.id for t in resultado.trechos],
                       "citadas": [f.id for f in r.fontes], "similaridade": round(resultado.similaridade_maxima, 4)})
    com = [x for x in linhas if not x["sem_resposta"]]
    sem = [x for x in linhas if x["sem_resposta"]]
    motivos = {m: sum(x["motivo"] == m for x in linhas) for m in sorted({x["motivo"] for x in linhas})}
    resumo = {"n": len(linhas), "corretas": sum(x["correta"] for x in linhas), "respondiveis_corretas": f"{sum(x['correta'] for x in com)}/{len(com)}",
              "recusas_corretas": f"{sum(x['correta'] for x in sem)}/{len(sem)}", "respondiveis_recusadas": sum(not x["sabe"] for x in com), "motivos": motivos}
    return {"resumo": resumo, "linhas": linhas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["busca", "respostas"])
    ap.add_argument("--conjunto", choices=["dev", "teste"], required=True)
    ap.add_argument("--modo", choices=["vetor", "hibrida"], default="hibrida")
    ap.add_argument("--limite", type=float)
    ap.add_argument("--sufixo", default="", help="para repetir a medida sem sobrescrever (o LLM varia entre execuções)")
    a = ap.parse_args()

    con = conectar()
    trechos = listar_trechos(con)
    embeddings = Embeddings()

    def criar(modo: str) -> Buscador:
        return Buscador(trechos, embeddings.de_pergunta, lambda v, n: buscar_por_vetor(con, v, n), modo)

    perguntas = ler_perguntas(a.conjunto)
    SAIDA.mkdir(parents=True, exist_ok=True)

    if a.etapa == "busca":
        resultado = {}
        for modo in ("vetor", "hibrida"):
            resultado[modo] = avaliar_busca(perguntas, criar(modo))
        resultado["limites"] = varrer_limites(perguntas, resultado["vetor"].pop("similaridades"))
        resultado["hibrida"].pop("similaridades")
        arquivo = SAIDA / f"busca_{a.conjunto}.json"
    else:
        if a.limite is None:
            ap.error("--limite é obrigatório na etapa de respostas")
        resultado = avaliar_respostas(perguntas, criar(a.modo), a.limite)
        arquivo = SAIDA / f"respostas_{a.conjunto}_{a.modo}{a.sufixo}.json"
    arquivo.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado.get("resumo", resultado), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
