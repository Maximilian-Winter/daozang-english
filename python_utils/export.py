"""
Simple Daozang Reconstructor
Takes JSON translations, outputs to English folder structure. Nothing else.
"""

import json
from pathlib import Path

# Category translations
TRANSLATIONS = {
    "dao": "daoist_canon",
    "fo": "buddhist_canon",
    "ru": "confucian_classics",
    "zi": "masters_philosophers",
    "ji": "literary_collections",
    "yi": "yijing_divination",
    "history": "historical_texts",
    "medicine": "medical_texts",
    "poem": "poetry_drama",
    "art": "arts_crafts",

    "正统道藏太平部": "taiping_section",
    "正统道藏太清部": "taiqing_section",
    "正统道藏太玄部": "taixuan_section",
    "正统道藏正一部": "zhengyi_section",
    "正统道藏洞玄部": "dongxuan_section",
    "正统道藏洞真部": "dongzhen_section",
    "正统道藏洞神部": "dongshen_section",
    "正统道藏续道藏": "continued_canon",
    "藏外": "extra_canonical",

    "众术类": "various_arts",
    "威仪类": "ceremonial_protocols",
    "戒律类": "precepts_disciplines",
    "方法类": "methods_techniques",
    "本文类": "primary_texts",
    "灵图类": "numinous_diagrams",
    "玉诀类": "jade_instructions",
    "神符类": "divine_talismans",
    "表奏类": "memorials_petitions",
    "记传类": "records_biographies",
    "谱箓类": "registers_rosters",
    "赞颂类": "praises_hymns",
}

def translate(name):
    """Translate or keep original if not in dict"""
    return TRANSLATIONS.get(name, name.lower().replace(" ", "_"))

def process(json_folder, output_folder):
    """Process all JSON files to English structure"""
    json_path = Path(json_folder)
    output_path = Path(output_folder)

    count = 0

    for json_file in json_path.glob("*.json"):
        # Load JSON
        data = json.load(open(json_file, 'r', encoding='utf-8'))

        # Get path from metadata
        source = data['metadata']['source_file'].replace("\\", "/")
        parts = source.split("/")

        # Translate path components
        english_parts = [translate(p) for p in parts[:-1]]
        filename = Path(parts[-1]).stem

        # Create output path
        out_dir = output_path / Path(*english_parts)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write markdown
        out_file = out_dir / f"{filename}.md"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"# {data['source_id']}\n\n")
            f.write(data['full_translation'])

        count += 1
        print(f"{count}: {out_file.relative_to(output_path)}")

    print(f"\nDone. {count} files.")

# Run it
process(
    json_folder="./translated_json",      # ← Change this
    output_folder="./daozang_translated"   # ← Change this
)