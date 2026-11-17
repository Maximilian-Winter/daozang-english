"""
Daozang Text Classifier using Decision Models (llama.cpp /v1/systemone endpoint)

Classifies translated Daozang texts across multiple dimensions:
- Primary tradition/school
- Core practices described
- Cosmological content
- Imagery patterns (fierce, celestial, transformative)
- Sky-dancing / dakini-parallel score
- Meditation and inner alchemy content

Requirements:
  - llama.cpp server running with a decision model
  - e.g.: llama serve -hf ggml-org/Kev-4B-GGUF
  - Translated Daozang texts as .txt or .md files in a directory

Usage:
  python daozang_classifier.py --input-dir ./daozang-english --output results.json
  python daozang_classifier.py --input-dir ./daozang-english --output results.csv --format csv
"""

import argparse
import json
import csv
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://localhost:8080"
ENDPOINT = "/v1/systemone"

# Maximum characters of text to send per request.
# Decision models have context limits; truncate very long texts.
MAX_STATE_CHARS = 12_000

# Retry / rate-limit settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# Question definitions — the heart of the classifier
# ---------------------------------------------------------------------------

QUESTIONS_PRIMARY = {
    # ── Tradition & School ──────────────────────────────────────────────
    "tradition": {
        "type": "choice",
        "instructions": "What is the primary Daoist tradition or school of this text?",
        "criteria": {
            "celestial_masters": "Tianshi / Zhengyi lineage, registers, talismans, communal ritual",
            "shangqing": "Shangqing / Maoshan, visualization, celestial journeys, revealed scriptures",
            "lingbao": "Lingbao, universal salvation, liturgy, influence from Buddhism",
            "quanzhen": "Quanzhen / Complete Perfection, inner alchemy, monasticism, asceticism",
            "neidan": "Neidan / internal alchemy, golden elixir, cinnabar fields, energy cultivation",
            "waidan": "Waidan / external alchemy, laboratory processes, mineral and herbal elixirs",
            "philosophical": "Laozi, Zhuangzi, Liezi tradition, cosmological or philosophical discourse",
            "fulu": "Talismanic, fu-lu registers, thunder rites, exorcism",
            "medical": "Medical, yangsheng, longevity practices, dietetics, sexual cultivation",
            "other": "Does not clearly fit any of the above categories",
        },
    },

    # ── Text Type ───────────────────────────────────────────────────────
    "text_type": {
        "type": "choice",
        "instructions": "What type of text is this?",
        "criteria": {
            "scripture": "Revealed scripture, canonical jing text",
            "commentary": "Commentary or exegesis on another text",
            "ritual_manual": "Ritual instructions, liturgical manual",
            "hagiography": "Biography of immortals, saints, or masters",
            "poetry": "Poetry, hymns, songs, or verse",
            "philosophical": "Philosophical treatise or discourse",
            "practical_manual": "Practical instructions for cultivation, meditation, or alchemy",
            "talismanic": "Talisman designs, fu-lu instructions, register texts",
            "narrative": "Stories, parables, dialogues, or narrative accounts",
            "encyclopedic": "Encyclopedic compilation or anthology",
        },
    },
}

QUESTIONS_CONTENT = {
    # ── Core Practices ──────────────────────────────────────────────────
    "meditation": {
        "type": "noul",
        "instructions": "Does this text describe meditation, contemplation, or stilling the mind?",
    },
    "visualization": {
        "type": "noul",
        "instructions": "Does this text describe visualization practices — seeing inner deities, lights, stars, or celestial palaces within the body or mind?",
    },
    "inner_alchemy": {
        "type": "noul",
        "instructions": "Does this text describe internal alchemy (neidan) — refining essence, qi, and spirit; cinnabar fields; the golden elixir; or embryonic breathing?",
    },
    "external_alchemy": {
        "type": "noul",
        "instructions": "Does this text describe external alchemy (waidan) — furnaces, minerals, cinnabar, lead, mercury, or laboratory processes?",
    },
    "talismanic_practice": {
        "type": "noul",
        "instructions": "Does this text describe talismans (fu), registers (lu), thunder rites, or exorcistic rituals?",
    },
    "breath_work": {
        "type": "noul",
        "instructions": "Does this text describe breathing practices — qi circulation, embryonic breathing, or breath retention?",
    },
    "body_cultivation": {
        "type": "noul",
        "instructions": "Does this text describe physical cultivation — daoyin exercises, gymnastics, massage, or sexual cultivation?",
    },
    "dietetics": {
        "type": "noul",
        "instructions": "Does this text describe dietary practices — fasting, abstaining from grains (bigu), consuming minerals, or herbal regimens?",
    },
    "ritual_liturgy": {
        "type": "noul",
        "instructions": "Does this text describe communal ritual, liturgy (zhai/jiao), offerings, or priestly ceremonies?",
    },

    # ── Cosmological Content ────────────────────────────────────────────
    "cosmology": {
        "type": "noul",
        "instructions": "Does this text describe cosmology — the creation of the universe, primordial chaos (hundun), the emergence of yin and yang, or the structure of heavens and hells?",
    },
    "celestial_bureaucracy": {
        "type": "noul",
        "instructions": "Does this text describe the celestial bureaucracy — heavenly officials, courts, ranks, registers, or petitions to celestial authorities?",
    },
    "immortals": {
        "type": "noul",
        "instructions": "Does this text describe immortals (xian), transcendents, or accounts of humans achieving immortality?",
    },
    "deities": {
        "type": "noul",
        "instructions": "Does this text describe specific deities — star gods, the Three Pure Ones, Jade Emperor, Queen Mother of the West, thunder gods, or other named divine beings?",
    },
}

QUESTIONS_IMAGERY = {
    # ── Imagery & Energy Patterns ───────────────────────────────────────
    "fierce_imagery": {
        "type": "score",
        "instructions": "How much fierce, wrathful, or intense imagery does this text contain — such as fire, destruction of demons, fierce deities, thunder, devouring, skull imagery, or violent transformation?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and intense"],
    },
    "sky_dancing": {
        "type": "score",
        "instructions": "How much sky-dancing, celestial flight, or aerial movement imagery does this text contain — such as flying through heavens, riding clouds, soaring, dancing among stars, ascending through celestial realms, or ecstatic movement through space?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
    "fire_transformation": {
        "type": "score",
        "instructions": "How much fire or heat transformation imagery does this text contain — such as refining by fire, inner fire, burning away impurities, the fire phases of alchemy, or transformative flames?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
    "light_rainbow": {
        "type": "score",
        "instructions": "How much light, radiance, or rainbow imagery does this text contain — such as golden light, five-colored clouds, radiant bodies, luminous emanations, or rainbow transformations?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
    "wild_unconventional": {
        "type": "score",
        "instructions": "How much wild, unconventional, or antinomian imagery does this text contain — such as disheveled hair, nakedness, drunkenness, breaking social norms, mad behavior as spiritual expression, or deliberate transgression?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
    "feminine_divine": {
        "type": "score",
        "instructions": "How much feminine divine imagery does this text contain — such as goddesses, divine women, female immortals, celestial maidens, the Queen Mother of the West, Mysterious Woman (Xuannü), or powerful feminine spiritual figures?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
    "emptiness_void": {
        "type": "score",
        "instructions": "How much emptiness or void imagery does this text contain — such as wu (nonbeing), xu (emptiness), the void, returning to the root, the uncarved block, or primordial nothingness?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
    "spontaneity_wuwei": {
        "type": "score",
        "instructions": "How much emphasis does this text place on spontaneity, naturalness, or wu-wei (non-action) — such as effortless action, following nature, non-interference, or the futility of deliberate striving?",
        "criteria": ["none", "minor mention", "moderate presence", "significant theme", "pervasive and central"],
    },
}

# Combined for a full scan
ALL_QUESTIONS = {**QUESTIONS_PRIMARY, **QUESTIONS_CONTENT, **QUESTIONS_IMAGERY}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextResult:
    """Classification result for one Daozang text."""
    file_path: str
    file_name: str
    char_count: int
    truncated: bool
    classification_time_ms: float
    answers: dict = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class DaozangClassifier:
    def __init__(
        self,
        server_url: str = DEFAULT_SERVER,
        model: Optional[str] = None,
        questions: Optional[dict] = None,
        max_chars: int = MAX_STATE_CHARS,
    ):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.questions = questions or ALL_QUESTIONS
        self.max_chars = max_chars
        self.endpoint = self.server_url + ENDPOINT

    def classify_text(self, text: str, file_path: str = "") -> TextResult:
        """Classify a single text. Returns a TextResult."""
        file_name = Path(file_path).name if file_path else ""
        char_count = len(text)
        truncated = char_count > self.max_chars

        if truncated:
            # Take beginning and end to capture both introductory framing
            # and concluding teachings, which often differ in character.
            half = self.max_chars // 2
            text = text[:half] + "\n\n[...]\n\n" + text[-half:]

        payload = {
            "state": text,
            "questions": self.questions,
        }
        if self.model:
            payload["model"] = self.model

        start = time.perf_counter()
        response_data = self._post(payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response_data is None:
            return TextResult(
                file_path=file_path,
                file_name=file_name,
                char_count=char_count,
                truncated=truncated,
                classification_time_ms=elapsed_ms,
                error="Request failed after retries",
            )

        return TextResult(
            file_path=file_path,
            file_name=file_name,
            char_count=char_count,
            truncated=truncated,
            classification_time_ms=elapsed_ms,
            answers=response_data.get("answers", {}),
        )

    def _post(self, payload: dict) -> Optional[dict]:
        """POST to the decision endpoint with retries."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
        return None

    def classify_directory(self, input_dir: str, extensions: tuple = (".txt", ".md")):
        """Yield TextResult for every matching file in input_dir (recursive)."""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            log.error("Input directory does not exist: %s", input_dir)
            return

        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        log.info("Found %d files to classify in %s", len(files), input_dir)

        for i, fpath in enumerate(files, 1):
            log.info("[%d/%d] Classifying: %s", i, len(files), fpath.name)
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    log.warning("  Skipping empty file: %s", fpath.name)
                    continue
                result = self.classify_text(text, file_path=str(fpath))
                if result.error:
                    log.error("  Error: %s", result.error)
                else:
                    log.info(
                        "  Done in %.0f ms — tradition: %s",
                        result.classification_time_ms,
                        _extract_choice(result.answers, "tradition"),
                    )
                yield result
            except Exception as e:
                log.error("  Exception processing %s: %s", fpath.name, e)
                yield TextResult(
                    file_path=str(fpath),
                    file_name=fpath.name,
                    char_count=0,
                    truncated=False,
                    classification_time_ms=0,
                    error=str(e),
                )


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def _extract_choice(answers: dict, key: str) -> str:
    """Get the chosen option from a choice answer."""
    answer = answers.get(key, {})
    return answer.get("choice", "unknown")


def _extract_noul(answers: dict, key: str) -> float:
    """Get the probability from a noul (yes/no) answer."""
    answer = answers.get(key, {})
    return answer.get("noul", 0.0)


def _extract_score(answers: dict, key: str) -> float:
    """Get the numeric score from a score answer."""
    answer = answers.get(key, {})
    return answer.get("score", 0.0)


def flatten_result(result: TextResult) -> dict:
    """Flatten a TextResult into a dict suitable for CSV export."""
    row = {
        "file_name": result.file_name,
        "file_path": result.file_path,
        "char_count": result.char_count,
        "truncated": result.truncated,
        "classification_time_ms": round(result.classification_time_ms, 1),
        "error": result.error or "",
    }

    answers = result.answers

    # Choice answers
    for key in ["tradition", "text_type"]:
        answer = answers.get(key, {})
        row[key] = answer.get("choice", "")
        row[f"{key}_confidence"] = round(answer.get("confidence", 0.0), 4)
        # Also store top probabilities
        probs = answer.get("probabilities", {})
        row[f"{key}_probabilities"] = json.dumps(probs) if probs else ""

    # Noul (yes/no) answers
    for key in [
        "meditation", "visualization", "inner_alchemy", "external_alchemy",
        "talismanic_practice", "breath_work", "body_cultivation", "dietetics",
        "ritual_liturgy", "cosmology", "celestial_bureaucracy", "immortals", "deities",
    ]:
        row[key] = round(_extract_noul(answers, key), 4)

    # Score answers
    for key in [
        "fierce_imagery", "sky_dancing", "fire_transformation", "light_rainbow",
        "wild_unconventional", "feminine_divine", "emptiness_void", "spontaneity_wuwei",
    ]:
        row[key] = round(_extract_score(answers, key), 4)
        # Store the max score for reference (number of levels minus 1)
        row[f"{key}_max"] = 4  # 5 levels, 0-indexed

    return row


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(results: list[TextResult], output_path: str):
    """Write results as JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s (%d results)", output_path, len(results))


def write_csv(results: list[TextResult], output_path: str):
    """Write results as a flat CSV."""
    if not results:
        log.warning("No results to write.")
        return

    rows = [flatten_result(r) for r in results]
    fieldnames = rows[0].keys()

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote CSV: %s (%d results)", output_path, len(results))


def write_report(results: list[TextResult], output_path: str):
    """Write a human-readable markdown report sorted by sky-dancing score."""
    scored = [
        (r, _extract_score(r.answers, "sky_dancing"))
        for r in results
        if r.error is None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    lines = [
        "# Daozang Classification Report",
        "",
        f"**Total texts classified:** {len(results)}",
        f"**Errors:** {sum(1 for r in results if r.error)}",
        "",
        "---",
        "",
        "## Top Texts by Sky-Dancing Score",
        "",
    ]

    for rank, (result, sky_score) in enumerate(scored[:50], 1):
        a = result.answers
        fierce = _extract_score(a, "fierce_imagery")
        feminine = _extract_score(a, "feminine_divine")
        wild = _extract_score(a, "wild_unconventional")
        fire = _extract_score(a, "fire_transformation")
        light = _extract_score(a, "light_rainbow")
        emptiness = _extract_score(a, "emptiness_void")
        wuwei = _extract_score(a, "spontaneity_wuwei")
        tradition = _extract_choice(a, "tradition")
        text_type = _extract_choice(a, "text_type")

        lines.extend([
            f"### {rank}. {result.file_name}",
            "",
            f"- **Tradition:** {tradition} | **Type:** {text_type}",
            f"- **Sky-Dancing:** {sky_score:.2f}/4 | **Fierce:** {fierce:.2f}/4 | **Feminine Divine:** {feminine:.2f}/4",
            f"- **Wild/Unconventional:** {wild:.2f}/4 | **Fire:** {fire:.2f}/4 | **Light/Rainbow:** {light:.2f}/4",
            f"- **Emptiness/Void:** {emptiness:.2f}/4 | **Wu-Wei:** {wuwei:.2f}/4",
            "",
        ])

    # Summary statistics for imagery scores
    lines.extend([
        "---",
        "",
        "## Imagery Score Distribution",
        "",
        "| Dimension | Texts > 2.0 | Texts > 3.0 | Max Score | Top Text |",
        "|-----------|-------------|-------------|-----------|----------|",
    ])

    for dimension in [
        "sky_dancing", "fierce_imagery", "feminine_divine", "wild_unconventional",
        "fire_transformation", "light_rainbow", "emptiness_void", "spontaneity_wuwei",
    ]:
        scores_with_names = [
            (_extract_score(r.answers, dimension), r.file_name)
            for r in results
            if r.error is None
        ]
        if not scores_with_names:
            continue
        scores_with_names.sort(key=lambda x: x[0], reverse=True)
        above_2 = sum(1 for s, _ in scores_with_names if s > 2.0)
        above_3 = sum(1 for s, _ in scores_with_names if s > 3.0)
        max_score, top_name = scores_with_names[0]
        label = dimension.replace("_", " ").title()
        lines.append(
            f"| {label} | {above_2} | {above_3} | {max_score:.2f} | {top_name} |"
        )

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Wrote report: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify Daozang texts using a llama.cpp decision model",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing translated Daozang texts (.txt / .md)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file path (.json, .csv, or .md for report)",
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "report", "all"], default=None,
        help="Output format (inferred from extension if not given)",
    )
    parser.add_argument(
        "--server", default=DEFAULT_SERVER,
        help=f"llama.cpp server URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model ID for multi-model server (optional)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=MAX_STATE_CHARS,
        help=f"Max characters per text (default: {MAX_STATE_CHARS})",
    )
    parser.add_argument(
        "--questions", choices=["all", "primary", "content", "imagery"],
        default="all",
        help="Which question set to run (default: all)",
    )
    parser.add_argument(
        "--extensions", default=".txt,.md",
        help="Comma-separated file extensions to include (default: .txt,.md)",
    )

    args = parser.parse_args()

    # Select question set
    question_sets = {
        "all": ALL_QUESTIONS,
        "primary": QUESTIONS_PRIMARY,
        "content": QUESTIONS_CONTENT,
        "imagery": QUESTIONS_IMAGERY,
    }
    questions = question_sets[args.questions]

    extensions = tuple(e.strip() for e in args.extensions.split(","))

    classifier = DaozangClassifier(
        server_url=args.server,
        model=args.model,
        questions=questions,
        max_chars=args.max_chars,
    )

    # Collect results
    results = list(classifier.classify_directory(args.input_dir, extensions=extensions))

    if not results:
        log.warning("No results collected. Check input directory and file extensions.")
        sys.exit(1)

    # Determine format
    fmt = args.format
    if fmt is None:
        ext = Path(args.output).suffix.lower()
        fmt = {".json": "json", ".csv": "csv", ".md": "report"}.get(ext, "json")

    # Write output
    output_base = Path(args.output).with_suffix("")

    if fmt == "all":
        write_json(results, str(output_base) + ".json")
        write_csv(results, str(output_base) + ".csv")
        write_report(results, str(output_base) + "_report.md")
    elif fmt == "json":
        write_json(results, args.output)
    elif fmt == "csv":
        write_csv(results, args.output)
    elif fmt == "report":
        write_report(results, args.output)

    # Print quick summary
    total = len(results)
    errors = sum(1 for r in results if r.error)
    total_time = sum(r.classification_time_ms for r in results)
    log.info("=" * 60)
    log.info("Classification complete: %d texts, %d errors, %.1fs total", total, errors, total_time / 1000)

    # Quick top-5 by sky-dancing
    scored = [
        (r, _extract_score(r.answers, "sky_dancing"))
        for r in results
        if r.error is None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored:
        log.info("")
        log.info("Top 5 by Sky-Dancing Score:")
        for r, s in scored[:5]:
            log.info("  %.2f  %s", s, r.file_name)


if __name__ == "__main__":
    main()
