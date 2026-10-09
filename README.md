# Collaborative Learning Analyzer

**Process-oriented analysis of small-group classroom discussions.**
Turns group-discussion audio/video into per-student participation and collaboration-quality
metrics — and, importantly, **refuses to invent numbers when a signal is missing.**

[![CI](https://github.com/firstxcst/collaborative-learning-analyzer/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/firstxcst/collaborative-learning-analyzer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](CHANGELOG.md)

[中文说明](README.zh-CN.md)

![Sample report](docs/images/report-sample.png)

> The report above is what `cla demo` produces from the **synthetic** fixture transcript shipped
> with this repository, analysed by the offline rule-based baseline. It demonstrates the output
> format and the "unavailable ≠ 0" behaviour — **not** analytic accuracy.
> See [Verification status](#verification-status).

---

## ⚠️ Read this before using it

This tool is currently **a teaching aid and an engineering prototype — not a measurement
instrument.** It must not be used to grade students, and its scores must not be reported as
research results.

Two reasons, stated plainly:

1. The metric weights have **not been empirically calibrated**, and no inter-rater agreement
   study (Cohen's Kappa / ICC) has been run.
2. The accuracy of the speech and vision pipelines has **never been evaluated on real classroom
   recordings**.

Both are listed as *unverified* in [Verification status](#verification-status). That section is
deliberately kept honest — that is the point of it.

---

## Table of contents

- [Try it in 60 seconds](#try-it-in-60-seconds)
- [What it does](#what-it-does)
- [Install](#install)
- [Usage](#usage)
- [Output](#output)
- [Scoring model](#scoring-model)
- [Privacy and data boundary](#privacy-and-data-boundary)
- [Verification status](#verification-status)
- [Project structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Try it in 60 seconds

**No API key. No `torch`. No `opencv`. Runs on Linux, macOS and Windows.**

```bash
git clone https://github.com/firstxcst/collaborative-learning-analyzer.git
cd collaborative-learning-analyzer
pip install -e .              # zero third-party dependencies
cla demo                      # runs the whole pipeline on the bundled fixture transcript
cla demo --html report.html   # self-contained HTML report (no CDN, opens offline)
```

`cla demo` uses the offline rule-based baseline on a synthetic transcript included in the repo.
It is a smoke test and a field-by-field demonstration, not an analysis of real students.

## What it does

| Modality | Extracted signals | Extra |
|---|---|---|
| **Speech** | who spoke, what was said, for how long (diarization + ASR) | `[audio]` `[diarization]` |
| **Semantics** | topic relevance, opinion collisions, argument depth, consensus quality, **per-speaker** semantic participation | `[llm]` or `[qwen]`, or the offline baseline |
| **Vision** | mutual attention, pointing gestures, group cohesion (pose + tracking) | `[vision]` |

These are fused into a 0–100 collaboration health score, one of five levels, per-student
contribution breakdowns, and actionable diagnoses.

**How members are identified.** The speaker roster comes from the audio side (who actually
talked). Vision track IDs (`person_*`) are **never** treated as members unless you supply an
explicit `member_mapping`. This prevents the classic failure mode of inventing "ghost members"
who never spoke.

---

## Install

The core package depends on **the standard library only** — installing it will not pull in
`torch` (~2 GB). Install the extra for the modality you need:

```bash
pip install -e .                 # core: scoring, data models, CLI, offline baseline
pip install -e ".[audio]"        # speech: Whisper ASR + audio probing
pip install -e ".[diarization]"  # speaker diarization (needs a HuggingFace token)
pip install -e ".[voiceprint]"   # voiceprint registration / identity alignment
pip install -e ".[vision]"       # vision: OpenCV + Ultralytics
pip install -e ".[llm]"          # semantics: any OpenAI-compatible endpoint
pip install -e ".[dev]"          # development: pytest / ruff / black / mypy
```

Everything at once (**large**, includes `torch`): `pip install -e ".[all]"`

Configuration lives in `.env` at the project root (resolved via `pyproject.toml`); environment
variables take precedence. Run `cla paths` to see the directories actually in use.

---

## Usage

### CLI

```bash
# Full pipeline
export LLM_PROVIDER=openai          # or dashscope / vllm / heuristic
export OPENAI_API_KEY=sk-...
cla analyze --audio discussion.wav --video discussion.mp4 \
    --members 4 --topic "your discussion topic" --output report.json

# Offline, no API key
cla analyze --transcript transcript.json --topic "wave, scattering, atmosphere" \
    --skip-video --output report.json

# HTML report
cla demo --html report.html
```

Useful flags: `--align-profiles` (voiceprint-based identity alignment),
`--annotated-video out.mp4`, `--skip-diarization` (loses per-student separation — the report
will say so explicitly).

### Already have a transcript?

Many classrooms use per-seat microphones or already produce subtitles. Import them and skip local
ASR entirely:

```json
{
  "source": "per-seat microphones",
  "topic_keywords": "wave, scattering, atmosphere",
  "segments": [
    {"speaker_id": "stu_A", "start": 0.0, "end": 3.2, "text": "..."}
  ]
}
```

### Python API

```python
from collaborative_learning_analyzer import analyze, render_report

report = analyze(
    audio_path="discussion.wav",
    video_path="discussion.mp4",
    topic="your discussion topic",
    group_id="group_3",
)
print(report.overall_health_score, report.health_level.value)
for c in report.individual_contributions:
    print(c.person_id, c.speaking_share, c.total_score, c.diagnosis["scored_dimensions"])

render_report(report, "report.html")
```

### Offline mode

`LLM_PROVIDER=heuristic` uses a rule-based baseline (discourse-marker counting). It is
**not an LLM** and does not claim LLM-level semantic understanding. When using it, pass the topic
as a **comma-separated keyword list** — topic relevance is then the share of utterances hitting
at least one keyword. Sentence-form topics are rejected (`null`) rather than approximated by
character overlap, because that proxy would silently feed a 0–100 score.

---

## Output

```jsonc
{
  "group_id": "group_3",
  "total_duration": 88.2,
  "member_ids": ["stu_A", "stu_B", "stu_C", "stu_D"],

  "participation_activity": 1.0,     // 0-1  actual talk time vs. target
  "evenness": 0.86,                  // 0-1  how evenly talk time is distributed
  "topic_relevance": 0.82,           // 0-1, or null when unavailable
  "interaction_depth": 0.71,         // 0-1, or null
  "turn_taking_pattern": "balanced", // balanced / monopolizing / chaotic / unknown
  "consensus_quality": 0.66,         // 0-1, or null
  "quality_score": 0.83,             // 0-1
  "overall_health_score": 83.2,      // 0-100
  "health_level": "good",            // critical / poor / fair / good / excellent

  "individual_contributions": [ /* per student: seconds, share, turns, 4 dims, total, notes */ ],
  "diagnoses": ["..."],
  "suggestions": ["..."],

  "data_completeness": { "audio": true, "video": false, "semantic": true },
  "warnings": ["Vision produced no data; the non-verbal dimension was excluded from scoring (it is not filled with a 0.5 default)."]
}
```

### Three things you must know

1. **Missing modalities are never fabricated.** If a modality produced no data, its dimensions are
   excluded from scoring and the weights are renormalised over the remaining dimensions. This is
   reported in `data_completeness` and `warnings`. A `0.5` never silently stands in for "unknown".
2. **`null` means "unavailable", not "zero".** `topic_relevance: null` means there was no valid
   semantic result — not that the group was off-topic.
3. **Floats are written at display precision** (scores to 1 decimal, ratios to 4).
   `save → load` preserves structure; floats round to that precision.

---

## Scoring model

```
activity  = min(1, total_speaking_time / (duration × target_talk_ratio))
evenness  = (1/HHI - 1) / (n - 1),  HHI = Σ pᵢ²,  pᵢ = speaking-time share
depth     = weighted mean(collision rate, argument depth, consensus quality)  # renormalised
quality   = weighted mean(evenness, topic_relevance, depth)                    # renormalised
health    = 100 × activity × quality
```

Key properties of `evenness`: perfectly equal shares → 1, one person monopolising → 0,
**nobody speaking → 0**.

The multiplicative activity gate means a silent group scores 0. In the previous design
(`evenness = 1 − 2σ`) a completely silent group scored **90.5/100 ("excellent")** — higher than a
healthy balanced group — and the `critical` level was mathematically unreachable.

Full derivation, the comparison with the broken v1.0.0 model, and an explicit statement that
**the weights are not yet calibrated** are in [`docs/SCORING_MODEL.md`](docs/SCORING_MODEL.md).

---

## Privacy and data boundary

Two things must be kept apart (the previous documentation was vague to the point of being wrong):

| Data | Leaves the machine? |
|---|---|
| Raw audio / video | **No.** ASR, diarization and pose estimation all run locally |
| The **transcript text** | **Yes** — it is sent to the LLM you configure (OpenAI by default). It is the necessary input for semantic analysis |
| Aggregated scores and diagnoses | Stored locally under `results/` |

So: **if you cannot send student dialogue text to a cloud service, use `LLM_PROVIDER=vllm`
against a locally hosted model, or `LLM_PROVIDER=heuristic` for the offline baseline.**
Transcripts contain real student speech — handle them according to your institution's research
ethics requirements.

The tool currently provides **no** data-lifecycle management (encryption at rest, retention
limits, automatic deletion). Earlier documentation promised these without implementing them;
this section now states the actual behaviour.

Obtain informed consent from students and guardians, and follow your institution's ethics review
requirements.

---

## Verification status

Verified and unverified are kept separate on purpose.

### Verified

- **Scoring logic** — bounds, monotonicity, reachability of all five levels, missing-modality
  handling, lossless round-trip. Asserted test by test in `tests/test_scoring.py`.
- **Packaging and imports** — installed top-level package name, lazy imports, no import-time side
  effects, CLI entry point.
- **Code hygiene** — unused imports and never-referenced definitions, enforced by **automated
  tests**.
- **End-to-end** — the full pipeline runs on **real media** produced by `tools/make_fixtures.py`.
- **Test effectiveness** — verified by **mutation testing**: 11 deliberate defects (including
  every bug from v1.0.0 restored) are all caught.
  See [`docs/MUTATION_TESTING.md`](docs/MUTATION_TESTING.md).

### Unverified — do not assume these work

- **Whisper ASR accuracy on Chinese classroom speech** — never evaluated on real recordings.
- **pyannote diarization accuracy under far-field, overlapping speech** — never evaluated.
- **YOLO person detection and gaze estimation accuracy** — monocular gaze is a hard problem; this
  implementation is a geometric heuristic that degrades badly in side views and under occlusion.
  **No accuracy data exists.**
- **Construct validity of the metrics** — weights uncalibrated, no inter-rater agreement study.

Going from "engineering demo" to "research instrument" requires: real classroom data →
independent double coding → human–machine agreement analysis → weight calibration and a validity
report.

### Roadmap

- [x] Rewrite the scoring model and add regression tests
- [x] Real-media fixtures and end-to-end verification
- [ ] Adaptation to real classrooms (far-field, multi-camera, noise)
- [ ] Agreement study against human coding (validity evidence)
- [ ] Web UI

---

## Project structure

```
collaborative-learning-analyzer/
├── src/                          # source (installed top-level package: collaborative_learning_analyzer)
│   ├── __init__.py               # lazy exports (importing does not pull cv2/torch)
│   ├── __main__.py               # python -m collaborative_learning_analyzer
│   ├── cli.py                    # the `cla` command
│   ├── config.py                 # configuration and path resolution (no import-time side effects)
│   ├── data_models.py            # data models and JSON round-trip
│   ├── audio_agent.py            # speech: diarization / ASR / voiceprint alignment / transcript import
│   ├── video_agent.py            # vision: pose / tracking / interaction inference
│   ├── semantic_agent.py         # semantics: windowed LLM analysis + offline baseline
│   ├── fusion_engine.py          # fusion and scoring
│   ├── pipeline.py               # end-to-end orchestration
│   ├── render.py                 # self-contained HTML report
│   └── py.typed                  # PEP 561 marker
├── tests/                        # test suite (see Verification status)
├── tools/make_fixtures.py        # generate real-media fixtures
├── tools/mutation_check.py       # mutation testing harness
├── tools/synth_tts.ps1           # Windows SAPI speech synthesis
├── examples/                     # two runnable examples
├── docs/                         # scoring model, audit remediation, mutation results
├── fixtures/                     # generated artifacts (media is not committed; see fixtures/README.md)
├── pyproject.toml                # single source of truth for dependencies and packaging
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE                       # AGPL-3.0 full text
```

---

## Development

```bash
pip install -e ".[all,dev]"

pytest                           # full suite
pytest -m "not slow"             # skip tests that need generated fixtures
python tools/mutation_check.py   # verify the tests still detect injected defects
ruff check src tests
black src tests && isort src tests
```

`pytest` must be green before committing. New scoring logic must come with an assertion that
**can actually fail** — this project verifies test effectiveness by mutation testing
(`docs/MUTATION_TESTING.md`).

---

## License

**GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)** — full text in
[LICENSE](LICENSE).

The practical consequence of AGPL: if you offer this software (or a modified version) to others as
a network service, you **must** provide them with the corresponding source code. If your use case
cannot accept that, do not use this software.

> Note: this repository previously contained four contradictory statements about its licence (an
> MIT badge in the README, AGPL in the README body, a hand-written summary in `LICENSE`, and MIT
> again in `CONTRIBUTING.md`). It is now uniformly AGPL-3.0-or-later with the official GNU text.
