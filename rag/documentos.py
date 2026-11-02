"""Transforma os documentos do restaurante em trechos pequenos, cada um com um id que pode ser citado."""
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "documentos"
NOMES_CATEGORIA = {"espetinhos": "Espetinhos", "acompanhamentos": "Acompanhamentos", "bebidas": "Bebidas"}


@dataclass(frozen=True)
class Trecho:
    id: str
    titulo: str
    texto: str


def normalizar(texto: str) -> str:
    """Tira acentos e caixa. Usada na busca por palavras e na conferência das respostas."""
    return unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()


def criar_slug(titulo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalizar(titulo)).strip("-")


def preco_em_reais(valor: float) -> str:
    return f"R$ {valor:.2f}".replace(".", ",")


def trechos_do_markdown(caminho: Path) -> list[Trecho]:
    """Cada seção "## " vira um trecho. O título entra no texto, porque ajuda a busca."""
    trechos = []
    for secao in re.split(r"(?m)^## ", caminho.read_text(encoding="utf-8"))[1:]:
        titulo, _, corpo = secao.partition("\n")
        titulo, corpo = titulo.strip(), corpo.strip()
        if not corpo:
            raise ValueError(f"{caminho.name}: seção vazia: {titulo}")
        trechos.append(Trecho(f"{caminho.stem}#{criar_slug(titulo)}", titulo, f"{titulo}. {corpo}"))
    return trechos


def trechos_do_cardapio(caminho: Path) -> list[Trecho]:
    """Um trecho por item e um resumo por categoria, para perguntas de comparação ("qual é o mais barato?")."""
    itens = json.loads(caminho.read_text(encoding="utf-8"))["itens"]
    trechos = []
    for item in itens:
        alergenos = ", ".join(item["alergenos"]) or "nenhum na receita"
        tipo = "É vegetariano." if item["vegetariano"] else "Não é vegetariano."
        texto = f"{item['nome']}: {preco_em_reais(item['preco'])}. {item['descricao']} {tipo} Alérgenos: {alergenos}."
        trechos.append(Trecho(f"cardapio#{item['id']}", item["nome"], texto))
    for categoria, nome in NOMES_CATEGORIA.items():
        lista = sorted((i for i in itens if i["categoria"] == categoria), key=lambda i: i["preco"])
        precos = "; ".join(f"{i['nome']}: {preco_em_reais(i['preco'])}" for i in lista)
        trechos.append(Trecho(f"cardapio#categoria-{categoria}", f"{nome}: todos os preços", f"{nome} do cardápio, do mais barato para o mais caro: {precos}."))
    return trechos


def carregar_trechos(pasta: Path = PASTA) -> list[Trecho]:
    trechos = trechos_do_cardapio(pasta / "cardapio.json")
    for caminho in sorted(pasta.glob("*.md")):
        trechos += trechos_do_markdown(caminho)
    ids = [t.id for t in trechos]
    if len(ids) != len(set(ids)):
        raise ValueError("há trechos com o mesmo id")
    return trechos
