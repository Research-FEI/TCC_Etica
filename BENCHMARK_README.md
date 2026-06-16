# Benchmark contra LLMs (ChatGPT e Gemini)

Reproduz, de forma versionada e auditável, a comparação do modelo do TCC contra dois
LLMs generalistas — o experimento das Tabelas I e II do artigo. Em vez de colar as
respostas na interface web e anotar notas à mão, este script envia o **mesmo prompt
padronizado** para a OpenAI e para o Gemini e grava as notas num CSV.

## Arquivos

- `benchmark_llm.py` — script principal.
- `prompts/benchmark_llm_prompt.md` — o prompt exato (coloque no apêndice do artigo).
- saída: `benchmark_results.csv` com as colunas
  `question_id, answer, professor_grade, model_tcc_grade, chatgpt_grade, gemini_grade`.

Coloque `benchmark_llm.py` e a pasta `prompts/` na raiz de `TCC-master/` (ao lado de
`api/`), pois os caminhos padrão apontam para `api/Service/data/questions.json` e
`api/results.csv`.

## 1. Instalar dependências

```bash
pip install openai google-genai
```

(`openai` só é necessário se for rodar o ChatGPT; `google-genai` só se for rodar o
Gemini.)

## 2. Obter as chaves

- **Gemini (grátis):** crie uma chave em https://aistudio.google.com/apikey — sem
  cartão de crédito. O free tier cobre os modelos Flash, que é o padrão do script.
- **OpenAI (centavos):** crie uma chave em https://platform.openai.com/api-keys.
  Precisa de uma forma de pagamento e ~US$5 de crédito, mas o custo real de avaliar
  algumas centenas de respostas curtas com `gpt-4o-mini` fica abaixo de US$0,10.

Defina como variáveis de ambiente (não coloque a chave no código):

```bash
export GEMINI_API_KEY="sua_chave_do_gemini"
export OPENAI_API_KEY="sua_chave_da_openai"
```

No Windows (PowerShell): `setx GEMINI_API_KEY "..."` e reabra o terminal.

## 3. Rodar

```bash
# teste pequeno: 3 respostas por questão, só Gemini (free), para validar tudo
python benchmark_llm.py --providers gemini --sample-per-question 3

# os dois provedores, amostra de 3 por questão (≈ o que o artigo reporta)
python benchmark_llm.py --sample-per-question 3

# dataset completo (210 respostas), os dois provedores
python benchmark_llm.py
```

Opções úteis:

| Flag | Para quê |
|------|----------|
| `--providers {both,openai,gemini}` | escolher qual(is) LLM(s) avaliar |
| `--sample-per-question N` | N respostas por questão (0 = todas) |
| `--limit N` | limita o total de respostas |
| `--openai-model` / `--gemini-model` | trocar o modelo/snapshot |
| `--temperature` | padrão 0.0 (determinístico) |
| `--sleep S` | pausa entre chamadas, se bater rate limit (ex.: `--sleep 1`) |

O script grava de forma incremental e **retoma de onde parou**: se rodar de novo, ele
pula as respostas já avaliadas e não chama a API à toa. Se uma chamada falhar de vez,
a célula fica vazia — basta rodar de novo para preencher só as que faltam.

## 4. Calcular as métricas

Depois de gerar `benchmark_results.csv`, as métricas de comparação (MAE, RMSE etc.)
saem com os scripts que já existem em `metrics/`, usando `professor_grade` como
referência e cada coluna de modelo como predição.

## 5. O que registrar no artigo (importante para reprodutibilidade)

- O nome **exato** do modelo e o *snapshot* usados (ex.: `gpt-4o-mini`,
  `gemini-2.5-flash`) — os modelos por trás dessas APIs mudam com o tempo.
- A **data** de execução.
- Temperatura (0.0) e o fato de a saída ter sido forçada a JSON.
- O prompt completo (em `prompts/benchmark_llm_prompt.md`), idêntico para os dois.

Com isso, a comparação deixa de ser um experimento manual irreprodutível e passa a
ser um procedimento que qualquer revisor consegue repetir.
