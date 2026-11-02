"""Mede as respostas com RAGAS: fidelidade, relevância da resposta e precisão do contexto.

Só entram as perguntas que o assistente respondeu. Quem julga é um LLM, então a nota é uma estimativa, e por isso o
acerto de fatos (avaliar.py) continua sendo a medida principal.
Uso: python -m avaliacao.ragas_avaliar --conjunto teste --modo hibrida
"""
import argparse
import json
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

from rag.embeddings import MODELO
from rag.resposta import MODELO_LLM

SAIDA = Path(__file__).resolve().parent.parent / "outputs" / "resultados"


def sem_citacoes(texto: str) -> str:
    return re.sub(r"\s*\[[^\[\]]+#[^\[\]]+\]", "", texto).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=["dev", "teste"], required=True)
    ap.add_argument("--modo", choices=["vetor", "hibrida"], required=True)
    a = ap.parse_args()

    respostas = json.loads((SAIDA / f"respostas_{a.conjunto}_{a.modo}.json").read_text(encoding="utf-8"))["linhas"]
    respondidas = [x for x in respostas if x["sabe"]]
    dados = EvaluationDataset.from_list([{"user_input": x["pergunta"], "response": sem_citacoes(x["resposta"]), "retrieved_contexts": x["contextos"]} for x in respondidas])
    llm = LangchainLLMWrapper(ChatAnthropic(model=MODELO_LLM, max_tokens=2000, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=MODELO, encode_kwargs={"normalize_embeddings": True}))
    metricas = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithoutReference()]
    tabela = evaluate(dados, metrics=metricas, llm=llm, embeddings=embeddings, run_config=RunConfig(max_workers=4, max_retries=5, timeout=120), show_progress=False).to_pandas()

    colunas = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy", "llm_context_precision_without_reference": "context_precision"}
    resumo = {"n_respondidas": len(respondidas), "n_total": len(respostas)}
    for coluna, nome in colunas.items():
        validos = tabela[coluna].dropna()
        resumo[nome] = round(float(validos.mean()), 4)
        resumo[f"{nome}_notas_validas"] = int(len(validos))
    (SAIDA / f"ragas_{a.conjunto}_{a.modo}.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
