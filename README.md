# daozang-english

An English translation of the *Zhengtong Daozang* (正統道藏), the orthodox Daoist canon — 1,676 texts, translated by machine, unreviewed by scholars.

---

## ⚠️ READ THIS BEFORE READING ANYTHING ELSE

### 1. These translations were produced by AI and have not been checked by a human expert.

Every text in this repository is a **first-pass machine translation**. No classicist, no Daoist practitioner, and no subject-matter specialist has reviewed the output. This means the files here will contain, with certainty rather than mere possibility:

- **Invented passages.** Language models produce fluent text even when they have not understood the source. A confident, readable paragraph here may correspond to nothing in the Chinese.
- **Systematically wrong technical vocabulary.** Alchemical and medical terminology is where this fails hardest, and it fails *consistently*, which makes it hard to spot. A real example found in these files: 黑錫 ("black tin") is the standard classical Chinese term for **lead**, but has been rendered throughout as "bismuth" — a different element with different toxicity. Related lead compounds (鉛霜, "lead frost") were shifted the same way. Assume other terms are similarly displaced.
- **Wrong units and durations.** 一伏時 (one full day–night, ~24 hours) has been glossed as "ten days" in places. Weights, measures, and time periods should not be trusted.
- **Lost register.** Ritual, allegorical, and literal-procedural passages read alike in translation. In the original they often do not.

**Do not cite this as a source.** Do not use it to settle a question of doctrine, history, or practice. It is a finding aid — a way to locate which of 1,676 texts might be worth having someone qualified actually read.

### 2. Many of these texts are recipes for poison.

External alchemy (外丹 *waidan*) is a large portion of this canon, and it is not metaphor. These are working laboratory procedures, written by people who ground, heated, and swallowed what they made. The materials they used are poisons, and identifying them by their English names does not make them safe:

| Text calls it | It is | What it does |
|---|---|---|
| 丹砂 / 硃砂 cinnabar, "divine cinnabar" | mercury(II) sulfide | Heating it releases **mercury vapour**, which is absorbed through the lungs and damages the brain, kidneys, and nervous system. Damage is cumulative and largely permanent. |
| 汞 / 水銀 mercury, "quicksilver," "mother mercury" | elemental mercury | Same. Vapour at room temperature; no smell, no warning. |
| 雄黃 realgar, 雌黃 orpiment, 砒霜 "frost," 信石 | arsenic sulfides and arsenic trioxide | Arsenic trioxide is lethal in quantities smaller than a pea. Chronic exposure causes nerve damage, skin lesions, and multiple cancers. |
| 黑錫 "black tin," 鉛霜 "lead frost," 黃丹, 密陀僧 | lead and lead compounds | Cumulative neurotoxin. No known safe exposure level. |
| 膽礬 chalcanthite | copper sulfate | Corrosive; damages liver and kidneys when ingested. |
| 烏頭 / 附子 aconite, "wutou" | *Aconitum* root | One of the most toxic plants in traditional pharmacy. Causes fatal cardiac arrhythmia. Deaths still occur from it today. |
| 巴豆 croton | *Croton tiglium* seed | Violent purgative; the oil is a severe blistering agent and tumour promoter. |

This is not a hypothetical concern. **Elixir poisoning killed several Tang dynasty emperors**, along with an unknown number of adepts and patrons, and the Chinese medical literature of the period discusses the symptoms — which the sufferers understood as signs of successful transformation.

Three specific hazards worth stating plainly:

- **Heating cinnabar in an unventilated space can cause acute mercury vapour poisoning.** You will not smell it. Symptoms may not appear for hours.
- **Some mineral preparations still sold today as traditional medicine or supplements contain cinnabar or realgar.** This is a live source of mercury and arsenic poisoning, documented in current medical case reports. Products marketed as traditional are not thereby safe.
- **Do not attempt to source, prepare, ingest, or experiment with any material described in these texts.** Not in reduced quantity, not "carefully," not to test whether it works. There is no safe way to run these procedures outside a licensed laboratory, and no reason to.

If you or someone else has ingested or been exposed to any of these substances, **contact emergency services or a poison control centre immediately.** Do not wait for symptoms.

---

## What is the Daozang?

The **Daozang** (道藏, "Treasury of the Dao") is the collected canon of Daoism, assembled over roughly fifteen centuries. The *Zhengtong* recension translated here was compiled in 1445. It contains:

- **Scriptures** presented as revelations from celestial beings
- **Alchemical treatises**, both external (*waidan*, laboratory) and internal (*neidan*, contemplative)
- **Ritual manuals** for ceremony, petition, and invocation
- **Philosophical texts** on the nature of the Dao
- **Hagiographies** of immortals and realized masters
- **Medical, pharmacological, and divinatory** material
- **Commentaries** spanning many dynasties

Most of this material has never been translated into any Western language. That is the reason this repository exists — not because machine translation is adequate to the task, but because for the great majority of these texts the alternative is nothing at all.

---

## Translation details

| | |
|---|---|
| **Texts** | 1,676 |
| **Source** | Chinese texts from https://github.com/wenyuange/dao |
| **Models** | Mistral Large 2512 + Mistral 14b |
| **Chunking** | 256 characters, with contextual overlap plus the full previously-translated chunk carried forward |
| **Segmentation** | semantic, punctuation-based, and marker-based |
| **Review status** | **None.** First-pass output. |

The chunking approach carries prior translated context forward to maintain terminological consistency within a text. Note that this propagates errors as readily as it propagates correct choices — a term mistranslated early will usually stay mistranslated throughout, which is how the lead/bismuth substitution above became systematic.

### Python utilities

`python_utils/` contains scripts to build a bilingual SQL database with a text-search interface, supporting search across both Chinese and English. Because the English is unreliable, **searching the Chinese and reading the English as a rough gloss is the more defensible workflow.**

---

## Further cautions on practice

Beyond the toxicology above, this canon contains material that assumes an initiated reader working under a teacher:

**Contemplative and internal practices.** Intensive meditation, breath retention, and energy-cultivation methods can precipitate real psychological and physiological harm — including anxiety, depersonalisation, sleep disruption, and in susceptible people psychotic episodes. These texts assume prerequisites they do not state, because they were not written for strangers.

**Ritual and talismanic material.** Written for practitioners within a transmission lineage, with assumed preparation, timing, and context that the text does not supply.

**Sexual cultivation texts.** These contain physical practices, some involving retention techniques, that are described without medical context.

**Pharmaceutical preparations.** See the warning above. Plant-based formulae are not automatically safer than mineral ones — aconite is a plant.

**Historical and military content.** Strategy, warfare divination, and descriptions of violence, presented in their period context.

### A reasonable way to approach this material

1. **Read it as literature and history**, which is what an unreviewed machine translation can honestly support.
2. **Check anything that matters against a scholarly translation.** Kristofer Schipper and Franciscus Verellen's *The Taoist Canon: A Historical Companion to the Daozang* is the standard reference for figuring out what a given text actually is.
3. **Find a qualified teacher** for any practice, in a living lineage.
4. **Assume the translation is wrong** wherever precision would matter.
5. **Respect the tradition** that preserved these texts for fifteen hundred years, including its own long-standing warnings about transmitting this material carelessly.

---

## Contributing

Corrections are welcome, and corrections to alchemical and pharmacological terminology are especially welcome — that is where the errors are both densest and most consequential. If you can read the classical Chinese and spot a mistranslated substance name, please open an issue. Flagging a misidentified poison is the single most useful contribution anyone can make to this repository.

---

## License and citation

The source texts are in the public domain. These translations are machine-generated and offered as-is, **without warranty of accuracy, completeness, or fitness for any purpose whatsoever** — explicitly including any practical, medicinal, alchemical, or ritual application.

If you reference this work, please state that the translations are unreviewed AI output.
