"""
Daozang Classification Results Analyzer

Reads the JSON output from daozang_classifier.py and produces detailed
analysis reports: top texts per dimension, directory breakdowns, cross-
dimension correlations, and interesting outliers.

Usage:
  python daozang_analyze.py results.json
  python daozang_analyze.py results.json --top 20
  python daozang_analyze.py results.json --output analysis_report.md
  python daozang_analyze.py results.json --output analysis.csv --csv
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Score extraction helpers
# ---------------------------------------------------------------------------

def get_choice(answers: dict, key: str) -> str:
    a = answers.get(key, {})
    return a.get("choice", "unknown")


def get_choice_confidence(answers: dict, key: str) -> float:
    a = answers.get(key, {})
    return a.get("confidence", 0.0)


def get_noul(answers: dict, key: str) -> float:
    a = answers.get(key, {})
    return a.get("noul", 0.0)


def get_score(answers: dict, key: str) -> float:
    a = answers.get(key, {})
    return a.get("score", 0.0)


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

SCORE_DIMENSIONS = [
    ("sky_dancing", "Sky-Dancing / Celestial Flight"),
    ("fierce_imagery", "Fierce / Wrathful Imagery"),
    ("feminine_divine", "Feminine Divine"),
    ("wild_unconventional", "Wild / Unconventional / Antinomian"),
    ("fire_transformation", "Fire Transformation"),
    ("light_rainbow", "Light / Rainbow"),
    ("emptiness_void", "Emptiness / Void"),
    ("spontaneity_wuwei", "Spontaneity / Wu-Wei"),
]

NOUL_DIMENSIONS = [
    ("meditation", "Meditation"),
    ("visualization", "Visualization"),
    ("inner_alchemy", "Inner Alchemy (Neidan)"),
    ("external_alchemy", "External Alchemy (Waidan)"),
    ("talismanic_practice", "Talismanic Practice"),
    ("breath_work", "Breathwork"),
    ("body_cultivation", "Body Cultivation"),
    ("dietetics", "Dietetics"),
    ("ritual_liturgy", "Ritual / Liturgy"),
    ("cosmology", "Cosmology"),
    ("celestial_bureaucracy", "Celestial Bureaucracy"),
    ("immortals", "Immortals"),
    ("deities", "Deities"),
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out errors
    return [r for r in data if r.get("error") is None]


def top_by_score(results: list[dict], key: str, n: int = 10) -> list[tuple[dict, float]]:
    scored = [(r, get_score(r["answers"], key)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def top_by_noul(results: list[dict], key: str, n: int = 10) -> list[tuple[dict, float]]:
    scored = [(r, get_noul(r["answers"], key)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def directory_averages(results: list[dict], key: str, is_score: bool = True) -> list[tuple[str, float, int]]:
    """Average score/noul per sub_dir. Returns [(dir, avg, count)]."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in results:
        d = r.get("sub_dir", "") or "(root)"
        val = get_score(r["answers"], key) if is_score else get_noul(r["answers"], key)
        buckets[d].append(val)
    out = []
    for d, vals in buckets.items():
        out.append((d, sum(vals) / len(vals), len(vals)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def tradition_averages(results: list[dict], key: str, is_score: bool = True) -> list[tuple[str, float, int]]:
    """Average score/noul per tradition classification."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in results:
        tradition = get_choice(r["answers"], "tradition")
        val = get_score(r["answers"], key) if is_score else get_noul(r["answers"], key)
        buckets[tradition].append(val)
    out = []
    for t, vals in buckets.items():
        out.append((t, sum(vals) / len(vals), len(vals)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def compute_composite_dakini_score(answers: dict) -> float:
    """Composite 'Dakini Energy' score combining relevant dimensions."""
    sky = get_score(answers, "sky_dancing")
    fierce = get_score(answers, "fierce_imagery")
    feminine = get_score(answers, "feminine_divine")
    wild = get_score(answers, "wild_unconventional")
    fire = get_score(answers, "fire_transformation")
    light = get_score(answers, "light_rainbow")
    # Weighted: sky-dancing and fierce are most dakini-relevant
    return (sky * 2.0 + fierce * 1.5 + feminine * 1.5 + wild * 1.0 + fire * 1.0 + light * 0.5) / 7.5


def find_outliers(results: list[dict]) -> list[tuple[dict, str, float]]:
    """Find texts that score unexpectedly high on a single dimension
    while being low on others — hidden gems."""
    outliers = []
    for r in results:
        a = r["answers"]
        scores = {key: get_score(a, key) for key, _ in SCORE_DIMENSIONS}
        if not scores:
            continue
        avg = sum(scores.values()) / len(scores)
        for key, val in scores.items():
            # High single score (>2.5) with low average (<1.5) = outlier
            if val > 2.5 and avg < 1.5:
                label = dict(SCORE_DIMENSIONS).get(key, key)
                outliers.append((r, label, val))
    outliers.sort(key=lambda x: x[2], reverse=True)
    return outliers


def cross_correlate(results: list[dict], key_a: str, key_b: str) -> list[tuple[dict, float, float]]:
    """Find texts that score high on BOTH dimensions."""
    pairs = []
    for r in results:
        a_val = get_score(r["answers"], key_a)
        b_val = get_score(r["answers"], key_b)
        pairs.append((r, a_val, b_val))
    # Sort by combined score
    pairs.sort(key=lambda x: x[1] + x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], top_n: int = 15) -> str:
    lines = []
    total = len(results)

    lines.append("# Daozang Classification Analysis Report")
    lines.append("")
    lines.append(f"**Total texts analyzed:** {total}")
    lines.append("")

    # ── Tradition distribution ──────────────────────────────────────
    lines.append("## Tradition Distribution")
    lines.append("")
    tradition_counts: dict[str, int] = defaultdict(int)
    for r in results:
        tradition_counts[get_choice(r["answers"], "tradition")] += 1
    lines.append("| Tradition | Count | % |")
    lines.append("|-----------|-------|---|")
    for t, c in sorted(tradition_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {t} | {c} | {c*100/total:.1f}% |")
    lines.append("")

    # ── Text type distribution ──────────────────────────────────────
    lines.append("## Text Type Distribution")
    lines.append("")
    type_counts: dict[str, int] = defaultdict(int)
    for r in results:
        type_counts[get_choice(r["answers"], "text_type")] += 1
    lines.append("| Type | Count | % |")
    lines.append("|------|-------|---|")
    for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {t} | {c} | {c*100/total:.1f}% |")
    lines.append("")

    # ── Composite Dakini Score Top N ────────────────────────────────
    lines.append(f"## 🔥 Top {top_n} by Composite Dakini Energy Score")
    lines.append("")
    lines.append("Weighted composite of sky-dancing (×2), fierce (×1.5), feminine divine (×1.5), wild (×1), fire (×1), light (×0.5).")
    lines.append("")
    dakini_scored = [(r, compute_composite_dakini_score(r["answers"])) for r in results]
    dakini_scored.sort(key=lambda x: x[1], reverse=True)
    lines.append("| # | File | Dir | Tradition | Dakini Score | Sky | Fierce | Feminine | Wild | Fire | Light |")
    lines.append("|---|------|-----|-----------|-------------|-----|--------|----------|------|------|-------|")
    for i, (r, ds) in enumerate(dakini_scored[:top_n], 1):
        a = r["answers"]
        lines.append(
            f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | {get_choice(a, 'tradition')} | "
            f"**{ds:.2f}** | {get_score(a, 'sky_dancing'):.2f} | {get_score(a, 'fierce_imagery'):.2f} | "
            f"{get_score(a, 'feminine_divine'):.2f} | {get_score(a, 'wild_unconventional'):.2f} | "
            f"{get_score(a, 'fire_transformation'):.2f} | {get_score(a, 'light_rainbow'):.2f} |"
        )
    lines.append("")

    # ── Top N per score dimension ───────────────────────────────────
    for key, label in SCORE_DIMENSIONS:
        lines.append(f"## Top {top_n}: {label}")
        lines.append("")
        lines.append("| # | File | Dir | Tradition | Score |")
        lines.append("|---|------|-----|-----------|-------|")
        for i, (r, s) in enumerate(top_by_score(results, key, top_n), 1):
            lines.append(
                f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | "
                f"{get_choice(r['answers'], 'tradition')} | {s:.2f} |"
            )
        lines.append("")

        # Tradition averages for this dimension
        lines.append(f"### {label} — Average by Tradition")
        lines.append("")
        lines.append("| Tradition | Avg Score | Texts |")
        lines.append("|-----------|-----------|-------|")
        for t, avg, cnt in tradition_averages(results, key, is_score=True):
            lines.append(f"| {t} | {avg:.2f} | {cnt} |")
        lines.append("")

        # Directory averages for this dimension (top 15)
        dir_avgs = directory_averages(results, key, is_score=True)
        if len(dir_avgs) > 1:
            lines.append(f"### {label} — Top Directories")
            lines.append("")
            lines.append("| Directory | Avg Score | Texts |")
            lines.append("|-----------|-----------|-------|")
            for d, avg, cnt in dir_avgs[:15]:
                lines.append(f"| {d} | {avg:.2f} | {cnt} |")
            lines.append("")

    # ── Top N per noul dimension ────────────────────────────────────
    lines.append("## Practice & Content Detection")
    lines.append("")
    lines.append("Probability that a text contains each practice/content type (noul = yes probability).")
    lines.append("")

    # Summary table
    lines.append("| Dimension | Texts > 0.8 | Texts > 0.5 | Avg |")
    lines.append("|-----------|-------------|-------------|-----|")
    for key, label in NOUL_DIMENSIONS:
        vals = [get_noul(r["answers"], key) for r in results]
        above_8 = sum(1 for v in vals if v > 0.8)
        above_5 = sum(1 for v in vals if v > 0.5)
        avg = sum(vals) / len(vals) if vals else 0
        lines.append(f"| {label} | {above_8} | {above_5} | {avg:.2f} |")
    lines.append("")

    # Individual top lists for noul dimensions
    for key, label in NOUL_DIMENSIONS:
        lines.append(f"### Top {top_n}: {label}")
        lines.append("")
        lines.append("| # | File | Dir | Tradition | Probability |")
        lines.append("|---|------|-----|-----------|-------------|")
        for i, (r, p) in enumerate(top_by_noul(results, key, top_n), 1):
            lines.append(
                f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | "
                f"{get_choice(r['answers'], 'tradition')} | {p:.3f} |"
            )
        lines.append("")

    # ── Cross-correlations (the fun part) ───────────────────────────
    lines.append("## 🔥 Cross-Dimension Highlights")
    lines.append("")

    # Sky-dancing + Fierce = Dao-Dakini
    lines.append(f"### Sky-Dancing × Fierce (Dao-Dakini Candidates)")
    lines.append("")
    lines.append("Texts scoring high on BOTH celestial flight and fierce/wrathful imagery.")
    lines.append("")
    lines.append("| # | File | Dir | Sky-Dancing | Fierce | Combined |")
    lines.append("|---|------|-----|-------------|--------|----------|")
    pairs = cross_correlate(results, "sky_dancing", "fierce_imagery")
    for i, (r, a_val, b_val) in enumerate(pairs[:top_n], 1):
        lines.append(
            f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | "
            f"{a_val:.2f} | {b_val:.2f} | {a_val + b_val:.2f} |"
        )
    lines.append("")

    # Feminine Divine + Fierce = Wrathful Goddess
    lines.append(f"### Feminine Divine × Fierce (Wrathful Goddess Candidates)")
    lines.append("")
    lines.append("| # | File | Dir | Feminine | Fierce | Combined |")
    lines.append("|---|------|-----|----------|--------|----------|")
    pairs = cross_correlate(results, "feminine_divine", "fierce_imagery")
    for i, (r, a_val, b_val) in enumerate(pairs[:top_n], 1):
        lines.append(
            f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | "
            f"{a_val:.2f} | {b_val:.2f} | {a_val + b_val:.2f} |"
        )
    lines.append("")

    # Wild + Emptiness = Antinomian Wisdom
    lines.append(f"### Wild/Unconventional × Emptiness (Antinomian Wisdom)")
    lines.append("")
    lines.append("| # | File | Dir | Wild | Emptiness | Combined |")
    lines.append("|---|------|-----|------|-----------|----------|")
    pairs = cross_correlate(results, "wild_unconventional", "emptiness_void")
    for i, (r, a_val, b_val) in enumerate(pairs[:top_n], 1):
        lines.append(
            f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | "
            f"{a_val:.2f} | {b_val:.2f} | {a_val + b_val:.2f} |"
        )
    lines.append("")

    # Sky-dancing + Feminine + Fire = Full Dakini Trinity
    lines.append(f"### Sky-Dancing × Feminine × Fire (Full Dakini Trinity)")
    lines.append("")
    lines.append("| # | File | Dir | Sky | Feminine | Fire | Total |")
    lines.append("|---|------|-----|-----|----------|------|-------|")
    trinity = []
    for r in results:
        a = r["answers"]
        s = get_score(a, "sky_dancing")
        f = get_score(a, "feminine_divine")
        fi = get_score(a, "fire_transformation")
        trinity.append((r, s, f, fi, s + f + fi))
    trinity.sort(key=lambda x: x[4], reverse=True)
    for i, (r, s, f, fi, total) in enumerate(trinity[:top_n], 1):
        lines.append(
            f"| {i} | {r['file_name']} | {r.get('sub_dir', '')} | "
            f"{s:.2f} | {f:.2f} | {fi:.2f} | {total:.2f} |"
        )
    lines.append("")

    # ── Outliers ────────────────────────────────────────────────────
    outliers = find_outliers(results)
    if outliers:
        lines.append("## 🔎 Outliers — Hidden Gems")
        lines.append("")
        lines.append("Texts with one unusually high dimension while otherwise low-scoring — potential hidden treasures.")
        lines.append("")
        lines.append("| File | Dir | High Dimension | Score | Tradition |")
        lines.append("|------|-----|----------------|-------|-----------|")
        for r, dim_label, val in outliers[:20]:
            lines.append(
                f"| {r['file_name']} | {r.get('sub_dir', '')} | {dim_label} | "
                f"{val:.2f} | {get_choice(r['answers'], 'tradition')} |"
            )
        lines.append("")

    # ── Directory heatmap ───────────────────────────────────────────
    lines.append("## Directory Heatmap")
    lines.append("")
    lines.append("Average scores per directory across all imagery dimensions.")
    lines.append("")

    header = "| Directory | Texts |"
    sep = "|-----------|-------|"
    for _, label in SCORE_DIMENSIONS:
        short = label.split("/")[0].split("(")[0].strip()[:12]
        header += f" {short} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # Collect all directories
    dir_texts: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        d = r.get("sub_dir", "") or "(root)"
        dir_texts[d].append(r)

    # Sort by composite average
    dir_composites = []
    for d, texts in dir_texts.items():
        composite = sum(compute_composite_dakini_score(r["answers"]) for r in texts) / len(texts)
        dir_composites.append((d, texts, composite))
    dir_composites.sort(key=lambda x: x[2], reverse=True)

    for d, texts, _ in dir_composites:
        row = f"| {d} | {len(texts)} |"
        for key, _ in SCORE_DIMENSIONS:
            avg = sum(get_score(r["answers"], key) for r in texts) / len(texts)
            row += f" {avg:.2f} |"
        lines.append(row)
    lines.append("")

    # ── Tradition heatmap ───────────────────────────────────────────
    lines.append("## Tradition Heatmap")
    lines.append("")
    lines.append("Average scores per tradition across all imagery dimensions.")
    lines.append("")

    lines.append(header.replace("Directory", "Tradition"))
    lines.append(sep)

    trad_texts: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        t = get_choice(r["answers"], "tradition")
        trad_texts[t].append(r)

    trad_composites = []
    for t, texts in trad_texts.items():
        composite = sum(compute_composite_dakini_score(r["answers"]) for r in texts) / len(texts)
        trad_composites.append((t, texts, composite))
    trad_composites.sort(key=lambda x: x[2], reverse=True)

    for t, texts, _ in trad_composites:
        row = f"| {t} | {len(texts)} |"
        for key, _ in SCORE_DIMENSIONS:
            avg = sum(get_score(r["answers"], key) for r in texts) / len(texts)
            row += f" {avg:.2f} |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


def generate_csv(results: list[dict]) -> str:
    """Generate a CSV with all scores flattened for spreadsheet analysis."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["file_name", "sub_path", "sub_dir", "tradition", "tradition_confidence",
              "text_type", "text_type_confidence", "char_count",
              "dakini_composite"]
    for key, _ in SCORE_DIMENSIONS:
        header.append(key)
    for key, _ in NOUL_DIMENSIONS:
        header.append(key)
    writer.writerow(header)

    # Rows
    for r in results:
        a = r["answers"]
        row = [
            r["file_name"],
            r.get("sub_path", ""),
            r.get("sub_dir", ""),
            get_choice(a, "tradition"),
            f"{get_choice_confidence(a, 'tradition'):.4f}",
            get_choice(a, "text_type"),
            f"{get_choice_confidence(a, 'text_type'):.4f}",
            r.get("char_count", 0),
            f"{compute_composite_dakini_score(a):.4f}",
        ]
        for key, _ in SCORE_DIMENSIONS:
            row.append(f"{get_score(a, key):.4f}")
        for key, _ in NOUL_DIMENSIONS:
            row.append(f"{get_noul(a, key):.4f}")
        writer.writerow(row)

    return output.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Daozang classification results",
    )
    parser.add_argument(
        "input",
        help="Path to results.json from daozang_classifier.py",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file (default: print to stdout). Use .md for report, .csv for spreadsheet.",
    )
    parser.add_argument(
        "--top", "-n", type=int, default=15,
        help="Number of top results per category (default: 15)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output as CSV instead of markdown report",
    )

    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} valid results.")

    if args.csv or (args.output and args.output.endswith(".csv")):
        content = generate_csv(results)
    else:
        content = generate_report(results, top_n=args.top)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote: {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
