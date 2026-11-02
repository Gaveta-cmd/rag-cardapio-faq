# Assistente de cardápio e FAQ de restaurante (RAG)

API que responde dúvidas de clientes usando só os documentos do restaurante: cardápio, horários, taxas, área de entrega, pagamento e trocas. Toda resposta cita a fonte. Quando os documentos não trazem a resposta, ela diz que não sabe.

O restaurante, o Espetinho da Esquina, é fictício e foi criado para este projeto.

> **Sobre o uso de IA:** este projeto foi desenvolvido com apoio de um assistente de IA (Claude Code), que escreveu o código, os documentos e o gabarito e rodou as avaliações, a partir do plano que defini.

```bash
curl -X POST http://127.0.0.1:8000/perguntar -H "Content-Type: application/json" \
  -d '{"pergunta": "Quanto custa a entrega no Jardim das Flores?"}'
```

```json
{
  "resposta": "A taxa de entrega no Jardim das Flores é de R$ 7,00 [entrega#taxas-de-entrega-por-bairro]. ...",
  "fontes": [{"id": "entrega#taxas-de-entrega-por-bairro", "titulo": "Taxas de entrega por bairro", "trecho": "..."}],
  "sabe": true,
  "motivo": "respondeu"
}
```

## Como funciona

```mermaid
flowchart LR
    A["Pergunta<br/>POST /perguntar"] --> B["Embedding<br/>multilingual-e5-small"]
    B --> C["pgvector<br/>trechos parecidos"]
    A --> D["BM25<br/>palavras iguais"]
    C --> E["Fusão por posição<br/>8 trechos"]
    D --> E
    C --> F{"Similaridade<br/>acima do limite?"}
    F -- não --> G["Não sei<br/>sem chamar o LLM"]
    F -- sim --> H["Claude Haiku<br/>responde citando a fonte"]
    E --> H
    H --> I{"Citou um trecho<br/>que veio da busca?"}
    I -- sim --> J["Resposta com fontes"]
    I -- não --> G
```

**Três barreiras contra respostas inventadas:** o limite de similaridade, a instrução para o LLM responder `NAO_SEI` e a conferência das citações. Uma resposta sem citação, ou citando um trecho que a busca não trouxe, nunca chega ao cliente.

## Resultados

São 50 perguntas com gabarito, commitadas antes da primeira execução: 20 de desenvolvimento, usadas para os ajustes, e 30 de teste, das quais 6 não têm resposta nos documentos. Uma resposta só conta como correta se trouxer os fatos esperados; uma pergunta sem resposta só conta se houver recusa.

**Busca: antes e depois da melhoria (teste, com 8 trechos)**

| Busca | Fonte certa entre os 8 trechos | MRR |
|---|---|---|
| Só vetor | 95,8% | 0,801 |
| Híbrida (BM25 + vetor) | 95,8% | 0,807 |

**Respostas completas (5 execuções do teste, porque o LLM varia)**

| Busca | Corretas de 30 (média) | Por execução | Recusas corretas de 6 |
|---|---|---|---|
| Só vetor | 26,8 | 27, 27, 27, 26, 27 | 6 em todas |
| **Híbrida** | **28,2** | 29, 28, 28, 28, 28 | 6 em todas |

**RAGAS (uma execução, só nas perguntas respondidas)**

| Busca | Respondidas | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|---|
| Só vetor | 21 de 30 | 1,000 | 0,928 | 0,848 |
| Híbrida | 23 de 30 | 0,925 | 0,927 | 0,830 |

### Como ler esses números

- **A melhoria é modesta.** A busca híbrida responde cerca de 1,4 pergunta a mais em 30. A busca em si empatou, e o RAGAS não mostra ganho: a fidelidade até cai. Mas as duas buscas responderam a conjuntos diferentes de perguntas, e a híbrida responde a casos mais difíceis, então as notas não são diretamente comparáveis.
- **O maior ganho veio de outro ajuste, no conjunto de desenvolvimento.** Passar de 5 para 8 trechos e permitir deduções diretas no prompt levou a híbrida de 16 para 19 acertos em 20. Isso é uma medida de desenvolvimento, não de teste.
- **O erro que sobra é recusar demais.** Nas 5 execuções da híbrida, todos os 9 erros vêm da conferência de citações barrando a resposta. Para um atendimento, prefiro esse erro a inventar uma regra, mas ele tem custo.
- **O limite de similaridade ajuda pouco.** Em 0,80, passam todas as perguntas com resposta, mas só 1 das 5 sem resposta é barrada no desenvolvimento, e nenhuma das 6 no teste. Quem recusa de fato é o LLM. Subir o limite cortaria perguntas boas.
- **A nota é otimista.** Os documentos, as perguntas e o gabarito têm o mesmo autor, e são só 49 trechos. O sistema nunca viu um cliente real.

## O que tem aqui

| Pasta/arquivo | Para quê |
|---|---|
| `documentos/` | Cardápio e regras do restaurante |
| `rag/` | Trechos, embeddings, banco, busca, resposta e API |
| `supabase/migrations/` | Tabela, RLS e função de busca, que valem para Supabase e Docker |
| `avaliacao/` | Perguntas, gabarito e medidas |
| `outputs/resultados/` | Respostas e números de cada execução |
| `tests/` | 25 testes, sem depender de banco, modelo ou API paga |
| `.github/workflows/ci.yml` | Lint e testes a cada envio |

## Como rodar

Com Docker:

```bash
cp .env.example .env   # preencha a ANTHROPIC_API_KEY
docker compose up -d --build
docker compose exec api python -m rag.indexar
```

A API fica em `http://127.0.0.1:8000`, com a documentação em `/docs`.

Testes e lint, sem banco, modelo nem chave:

```bash
pip install -r requirements-ci.txt
ruff check .
pytest -q
```

Avaliação, que gasta alguns centavos de API:

```bash
pip install -r requirements-avaliacao.txt
python -m avaliacao.avaliar busca --conjunto teste
python -m avaliacao.avaliar respostas --conjunto teste --modo hibrida --limite 0.80
python -m avaliacao.ragas_avaliar --conjunto teste --modo hibrida
```

## Limitações

- **Ainda não há demo online nem instância no Supabase.** O banco testado é Postgres com pgvector no Docker. A migração foi escrita para valer também no Supabase, mas não foi executada lá.
- **O CI ainda não rodou no GitHub.** Os mesmos comandos passaram localmente, em um ambiente limpo.
- **A imagem é grande** (3,6 GB), por causa do PyTorch e do modelo de embeddings dentro dela.
- **Respostas não são determinísticas:** o modelo usado não aceita o parâmetro `temperature`.
- **O BM25 fica em memória** e é montado quando a API sobe. Isso serve para 49 trechos, não para uma base grande.
