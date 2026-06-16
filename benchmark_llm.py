#!/usr/bin/env python3
"""
benchmark_llm.py
================
Compara o modelo especializado do TCC contra LLMs generalistas (ChatGPT/OpenAI e
Gemini/Google) na avaliação automática de questões dissertativas do ENADE.

O script reaproveita os dados que já existem no repositório:
  - api/Service/data/questions.json  -> enunciado, gabarito e palavras-chave
  - api/results.csv                  -> resposta do aluno, nota do professor
                                        (original_grade) e nota do modelo TCC
                                        (evaluated_grade)

Para cada resposta, envia o MESMO prompt padronizado para a OpenAI e para o Gemini,
extrai a nota (escala 0,00-1,00) e grava um CSV no mesmo formato das Tabelas I e II
do artigo:

  question_id, answer, professor_grade, model_tcc_grade, chatgpt_grade, gemini_grade

Características:
  - O prompt é idêntico para os dois provedores (comparação justa).
  - Temperatura 0.0 por padrão (reprodutibilidade).
  - Chaves de API lidas de variáveis de ambiente (NUNCA hardcoded).
  - Retentativa com backoff exponencial em erros de rate limit/transitórios.
  - Retomada (resume): grava incrementalmente e pula respostas já avaliadas,
    para não pagar/chamar a API duas vezes em re-execuções.

Uso rápido (instalar dependências e definir as chaves antes — ver BENCHMARK_README.md):

  # roda os dois provedores em uma amostra de 3 respostas por questão (como o artigo)
  python benchmark_llm.py --sample-per-question 3

  # roda apenas o Gemini (free tier) em todo o dataset
  python benchmark_llm.py --providers gemini

  # roda tudo
  python benchmark_llm.py
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Prompt padronizado (idêntico aos dois provedores). Mantido em sincronia com
# prompts/benchmark_llm_prompt.md — se editar aqui, edite lá também.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Você é um avaliador especialista de questões dissertativas do ENADE (Exame "
    "Nacional de Desempenho dos Estudantes). Sua tarefa é atribuir uma nota "
    "numérica contínua à resposta de um aluno.\n\n"
    "Critérios obrigatórios de correção:\n"
    "- A nota deve estar na escala de 0,00 a 1,00 (use ponto decimal e duas casas).\n"
    "- A avaliação baseia-se na aderência ao padrão de resposta (gabarito) fornecido "
    "e na cobertura dos conceitos e nomenclaturas obrigatórios indicados.\n"
    "- NÃO atribua nota alta a textos bem escritos do ponto de vista gramatical ou "
    "estrutural que fujam parcial ou totalmente do conteúdo técnico exigido.\n"
    "- Seja rigoroso e consistente, como um corretor humano experiente seguindo a "
    "rubrica.\n\n"
    'Responda EXCLUSIVAMENTE com um objeto JSON, sem nenhum texto adicional, no '
    'formato:\n{"nota": <número entre 0 e 1>}'
)

USER_TEMPLATE = (
    "ENUNCIADO DA QUESTÃO:\n{question}\n\n"
    "PADRÃO DE RESPOSTA (GABARITO):\n{reference_answers}\n\n"
    "PALAVRAS-CHAVE OBRIGATÓRIAS:\n{keywords}\n\n"
    "RESPOSTA DO ALUNO A SER AVALIADA:\n{student_answer}\n\n"
    "Atribua a nota seguindo estritamente os critérios. Responda apenas com o JSON."
)


def build_user_prompt(question: dict, student_answer: str) -> str:
    refs = question.get("reference_answer", [])
    refs_block = "\n".join(f"{i+1}. {r}" for i, r in enumerate(refs))
    keywords = ", ".join(question.get("keywords", []))
    return USER_TEMPLATE.format(
        question=question.get("question", "").strip(),
        reference_answers=refs_block,
        keywords=keywords,
        student_answer=student_answer.strip(),
    )


# --------------------------------------------------------------------------- #
# Parsing robusto da nota retornada pelo modelo
# --------------------------------------------------------------------------- #
def parse_score(text: str):
    """Extrai a nota (0..1) da resposta do modelo. Retorna float ou None."""
    if not text:
        return None
    text = text.strip()
    # 1) tenta JSON direto
    for candidate in (text, _strip_code_fences(text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                for key in ("nota", "score", "grade", "value"):
                    if key in obj:
                        return _clamp(float(obj[key]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # 2) fallback: primeiro número que pareça uma nota
    m = re.search(r'"?nota"?\s*[:=]\s*([01](?:[.,]\d+)?)', text, re.IGNORECASE)
    if m:
        return _clamp(float(m.group(1).replace(",", ".")))
    m = re.search(r'\b([01](?:[.,]\d+)?)\b', text)
    if m:
        val = float(m.group(1).replace(",", "."))
        return _clamp(val)
    # 3) talvez escala 0-100
    m = re.search(r'\b(\d{1,3})\b', text)
    if m:
        return _clamp(float(m.group(1)) / 100.0)
    return None


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())


def _clamp(v: float) -> float:
    return round(max(0.0, min(1.0, v)), 2)


# --------------------------------------------------------------------------- #
# Retentativa com backoff
# --------------------------------------------------------------------------- #
def with_retries(fn, max_attempts=5, base_delay=2.0, label=""):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — queremos capturar erros de SDK/rede
            transient = any(
                s in str(exc).lower()
                for s in ("rate", "429", "quota", "timeout", "503", "overloaded",
                          "temporarily", "connection")
            )
            if attempt == max_attempts or not transient:
                print(f"  [{label}] erro definitivo: {exc}", file=sys.stderr)
                return None
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  [{label}] tentativa {attempt} falhou ({exc}); "
                  f"aguardando {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)
    return None


# --------------------------------------------------------------------------- #
# Provedores
# --------------------------------------------------------------------------- #
class OpenAIGrader:
    def __init__(self, model: str, temperature: float):
        from openai import OpenAI  # import tardio: só carrega se for usar
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def grade(self, system_prompt: str, user_prompt: str):
        def _call(use_temperature=True):
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            if use_temperature:
                kwargs["temperature"] = self.temperature
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        def _call_safe():
            try:
                return _call(use_temperature=True)
            except Exception as exc:  # alguns modelos novos rejeitam temperature/format
                if "temperature" in str(exc).lower() or "response_format" in str(exc).lower():
                    return _call(use_temperature=False)
                raise

        text = with_retries(_call_safe, label="openai")
        return parse_score(text), text


class GeminiGrader:
    def __init__(self, model: str, temperature: float):
        from google import genai  # SDK google-genai (novo)
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Defina a variável de ambiente GEMINI_API_KEY (ou GOOGLE_API_KEY)."
            )
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model
        self.temperature = temperature

    def grade(self, system_prompt: str, user_prompt: str):
        def _call():
            config = self.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json",
            )
            resp = self.client.models.generate_content(
                model=self.model, contents=user_prompt, config=config
            )
            return resp.text

        text = with_retries(_call, label="gemini")
        return parse_score(text), text


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
OUTPUT_COLUMNS = [
    "question_id", "answer", "professor_grade", "model_tcc_grade",
    "chatgpt_grade", "gemini_grade",
]


def row_key(question_id, answer) -> str:
    h = hashlib.sha1(answer.strip().encode("utf-8")).hexdigest()[:16]
    return f"{question_id}::{h}"


def load_existing(output_path: Path) -> dict:
    """Carrega resultados já gravados para permitir retomada."""
    if not output_path.exists():
        return {}
    done = {}
    with output_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[row_key(row["question_id"], row["answer"])] = row
    return done


def load_questions(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data["questions"] if isinstance(data, dict) else data
    return {int(q["id"]): q for q in qs}


def load_answers(results_csv: Path):
    rows = []
    with results_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "question_id": int(row["question_id"]),
                "answer": row["answer"],
                "professor_grade": row.get("original_grade", ""),
                "model_tcc_grade": row.get("evaluated_grade", ""),
            })
    return rows


def sample_rows(rows, per_question, limit, seed):
    rng = random.Random(seed)
    if per_question:
        by_q = {}
        for r in rows:
            by_q.setdefault(r["question_id"], []).append(r)
        out = []
        for qid in sorted(by_q):
            bucket = by_q[qid][:]
            rng.shuffle(bucket)
            out.extend(bucket[:per_question])
        rows = out
    if limit:
        rows = rows[:limit]
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="api/Service/data/questions.json",
                    help="caminho para questions.json")
    ap.add_argument("--input", default="api/results.csv",
                    help="CSV com respostas + notas (question_id, answer, "
                         "original_grade, evaluated_grade)")
    ap.add_argument("--output", default="benchmark_results.csv",
                    help="CSV de saída")
    ap.add_argument("--providers", default="both",
                    choices=["both", "openai", "gemini"],
                    help="quais LLMs avaliar")
    ap.add_argument("--openai-model", default="gpt-4o-mini",
                    help="modelo da OpenAI (confirme o snapshot atual)")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash",
                    help="modelo do Gemini (free tier cobre os Flash)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sample-per-question", type=int, default=0,
                    help="avalia N respostas por questão (0 = todas)")
    ap.add_argument("--limit", type=int, default=0,
                    help="limita o total de respostas (0 = sem limite)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pausa em segundos entre chamadas (ajuda no rate limit)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    q_path = Path(args.questions)
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not q_path.exists() or not in_path.exists():
        sys.exit(f"Arquivo não encontrado. Verifique --questions ({q_path}) e "
                 f"--input ({in_path}). Rode a partir da pasta TCC-master.")

    questions = load_questions(q_path)
    rows = load_answers(in_path)
    rows = sample_rows(rows, args.sample_per_question, args.limit, args.seed)

    use_openai = args.providers in ("both", "openai")
    use_gemini = args.providers in ("both", "gemini")

    openai_grader = OpenAIGrader(args.openai_model, args.temperature) if use_openai else None
    gemini_grader = GeminiGrader(args.gemini_model, args.temperature) if use_gemini else None

    existing = load_existing(out_path)
    print(f"{len(rows)} respostas selecionadas | "
          f"{len(existing)} já no arquivo de saída (serão reaproveitadas).")
    print(f"Provedores: openai={use_openai} ({args.openai_model}) "
          f"gemini={use_gemini} ({args.gemini_model}) | temp={args.temperature}\n")

    results = []
    api_calls = 0
    for i, r in enumerate(rows, 1):
        qid, answer = r["question_id"], r["answer"]
        key = row_key(qid, answer)
        prev = existing.get(key, {})
        out = {
            "question_id": qid,
            "answer": answer,
            "professor_grade": r["professor_grade"],
            "model_tcc_grade": r["model_tcc_grade"],
            "chatgpt_grade": prev.get("chatgpt_grade", ""),
            "gemini_grade": prev.get("gemini_grade", ""),
        }

        question = questions.get(qid)
        if question is None:
            print(f"[{i}/{len(rows)}] q{qid}: questão não encontrada, pulando.")
            results.append(out)
            continue

        system_prompt = SYSTEM_PROMPT
        user_prompt = build_user_prompt(question, answer)

        if use_openai and not _has_value(out["chatgpt_grade"]):
            score, _ = openai_grader.grade(system_prompt, user_prompt)
            out["chatgpt_grade"] = "" if score is None else f"{score:.2f}"
            api_calls += 1
            if args.sleep:
                time.sleep(args.sleep)

        if use_gemini and not _has_value(out["gemini_grade"]):
            score, _ = gemini_grader.grade(system_prompt, user_prompt)
            out["gemini_grade"] = "" if score is None else f"{score:.2f}"
            api_calls += 1
            if args.sleep:
                time.sleep(args.sleep)

        results.append(out)
        print(f"[{i}/{len(rows)}] q{qid} | prof={out['professor_grade']} "
              f"tcc={out['model_tcc_grade']} "
              f"chatgpt={out['chatgpt_grade'] or '-'} "
              f"gemini={out['gemini_grade'] or '-'}")

        # grava incrementalmente a cada 5 itens (e ao final) -> retomada segura
        if i % 5 == 0:
            write_output(out_path, results, existing)

    write_output(out_path, results, existing)
    print(f"\nConcluído. {api_calls} chamadas de API feitas nesta execução.")
    print(f"Resultados em: {out_path.resolve()}")


def _has_value(v) -> bool:
    return v not in (None, "", "nan")


def write_output(out_path: Path, results, existing):
    """Mescla resultados desta execução com os já existentes e grava o CSV."""
    merged = dict(existing)
    for out in results:
        merged[row_key(out["question_id"], out["answer"])] = out
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for out in merged.values():
            w.writerow({k: out.get(k, "") for k in OUTPUT_COLUMNS})


if __name__ == "__main__":
    main()
