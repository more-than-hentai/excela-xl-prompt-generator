#!/usr/bin/env python3
"""
Scenario-based prompt maker for Qwen-Image (cuts/shots + keyword bundles).

요약
- 시나리오(상황/이야기)를 입력으로 받아 컷(샷) 프리셋별로 프롬프트를 생성합니다.
- Qwen-Image 공식 가이드라인 형식을 따르며(문장/라벨/태그), 로컬 Ollama LLM을 호출합니다.
- 의상/인물 프리셋 번들을 선택적으로 시드에 덧붙일 수 있습니다.
- 각 컷별 결과를 개별 파일로 저장합니다: output/scenarios/<slug>/<NN>_<shot>.txt

의존
- scripts/generate_prompts.py 의 Qwen-Image 유틸리티를 import 하여 재사용합니다.

사용 예시
  python3 scripts/scenario_prompt_maker.py \
    --scenario "Rainy neon alley; a poised woman exits a jazz bar" \
    --bundle quality-outfit --bundle quality-character \
    --style tags --variants 3 --model "qwen2.5:7b-instruct-q5_K_M"

  # 시스템 프롬프트를 QWEN 가이드 파일로 고정(기본값)
  python3 scripts/scenario_prompt_maker.py \
    --scenario "Dawn on a foggy harbor, minimal elegance" \
    --style structured --variants 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import hashlib
from typing import Dict, List, Tuple

# 동경: 동일 디렉토리의 generate_prompts를 import (repo 루트에서 실행 권장)
try:
    from scripts.generate_prompts import (
        iter_qwen_image_prompts,
        sanitize_seed,
        ollama_chat,
        ollama_generate,
        build_qwen_chatml,
    )
except Exception:  # Fallback: 같은 폴더에서 직접 import
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from generate_prompts import (  # type: ignore
        iter_qwen_image_prompts,
        sanitize_seed,
        ollama_chat,
        ollama_generate,
        build_qwen_chatml,
    )


def slugify(text: str, max_len: int = 80, add_hash: bool = True) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-") or "scenario"
    if max_len and len(slug) > max_len:
        if add_hash:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
            head_len = max(10, max_len - 1 - len(digest))
            slug = slug[:head_len].rstrip("-") + "-" + digest
        else:
            slug = slug[:max_len].rstrip("-")
    return slug


def default_bundles() -> Dict[str, List[str]]:
    """Predefined keyword bundles.

    - quality-outfit: 고품질 의상/룩 디테일 중심
    - quality-character: 고품질 인물/태도/메이크업 중심
    - nsfw-soft: 비노골적(NSFW-leaning) 소프트 톤(비노출, 란제리/부드러운 조명)
    - nsfw-boudoir: 부도와르 무드(비노골적, 친밀한 조명/무드, 실루엣 중심)
    """
    return {
        "quality-outfit": [
            "couture outfit",
            "tailored silhouette",
            "asymmetric drape",
            "structured corset",
            "hand-sewn lace applique",
            "micro-bead accents",
            "satin finish",
            "sheer overlay",
            "precise seam work",
            "minimal metallic accessories",
        ],
        "quality-character": [
            "elegant adult woman",
            "photorealistic skin detail",
            "subtle pore texture",
            "catchlight in eyes",
            "refined posture",
            "sleek hairstyle",
            "soft contour makeup",
            "glossy lips",
            "minimal jewelry",
            "graceful presence",
        ],
        # Non-explicit NSFW-leaning bundles (keep tasteful and adult-only)
        "nsfw-soft": [
            "tasteful lingerie",
            "silk robe",
            "soft boudoir lighting",
            "low key lighting",
            "subtle skin highlight",
            "glamour tone",
            "private setting",
            "suggestive posing",
        ],
        "nsfw-boudoir": [
            "boudoir session",
            "luxury bedroom set",
            "sheer fabric",
            "lace bodysuit",
            "backlighting silhouette",
            "film grain",
            "intimate mood",
            "soft focus bokeh",
        ],
    }


def shot_preset(name: str) -> List[Tuple[str, str]]:
    """Return a list of (shot_key, extra_tokens) for a preset.

    Each extra_tokens is a concise, comma-separated hint bundle in English.
    """
    key = (name or "").lower()
    if key in ("storyboard", "default", "std"):
        return [
            (
                "01_establishing",
                "establishing shot, wide angle, high vantage, environment context, 24mm lens, rule of thirds, cinematic lighting",
            ),
            (
                "02_wide",
                "full-body wide shot, 35mm lens, spatial context, balanced composition, natural pose",
            ),
            (
                "03_medium",
                "medium shot (waist-up), 50mm lens, conversational framing, subtle background depth",
            ),
            (
                "04_closeup",
                "tight close-up, 85mm lens, shallow depth of field, skin texture, catchlight",
            ),
            (
                "05_over_shoulder",
                "over-the-shoulder shot, subject-of-interest in focus, narrative framing, eye-level",
            ),
            (
                "06_detail",
                "macro detail, 100mm macro, texture focus, intricate material, soft bokeh",
            ),
        ]
    # Fallback: single generic shot
    return [("01_scene", "balanced composition, cinematic lighting, realistic detail")]


def read_text_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def build_story_prompt(topics: List[str], sentences: int, language: str, style: str) -> Tuple[str, str]:
    """Return (system, user) to generate a concise scenario that includes topics.

    language: 'en' or 'ko'
    style: 'logline' | 'vignette'
    """
    lang_label = "English" if language.lower().startswith("en") else "Korean"
    style_hint = "cinematic logline" if style == "logline" else "short cinematic vignette"
    system = (
        "You are a creative scenarist who writes concise, visual-first scenarios. "
        "Always include the provided topics explicitly. Avoid meta comments and lists."
    )
    topics_str = ", ".join(t for t in topics if t.strip())
    user = (
        f"Write exactly {sentences} sentences in {lang_label} as a {style_hint}. "
        f"Include these topics: {topics_str}. "
        "Keep it concrete and evocative; no dialogue unless essential. Output one paragraph only."
    )
    return system, user


def build_shotlist_prompt(
    scenario: str,
    topics: List[str],
    num_cuts: int,
    duration_sec: int,
    language: str,
) -> Tuple[str, str]:
    """Return (system, user) for generating a concise shot list.

    Output format: exactly N lines, each using ' | ' as a field separator.
    Recommended fields per line: Shot <#>: <short-name> | duration: <sec> | shot: <type/angle> | lens: <focal> | camera: <movement/height> | subject/action: <...> | continuity: <anchor>
    """
    lang_label = "English" if language.lower().startswith("en") else "Korean"
    system = (
        "You are a cinematographer planning a tightly-edited micro video. "
        "Produce a coherent shot list with temporal continuity and visual progression. "
        "Use concise film language."
    )
    tops = ", ".join(t for t in topics if t.strip())
    user = (
        f"Scenario (context): {scenario.strip()}\n"
        f"Topics: {tops if tops else '(none)'}\n"
        f"Total duration: ~{int(duration_sec)} seconds. Shots: exactly {int(num_cuts)}.\n"
        f"Write in {lang_label}. Output exactly {int(num_cuts)} lines. "
        "Each line format: Shot <#>: <short-name> | duration: <sec> | shot: <type/angle> | lens: <focal> | camera: <movement/height> | subject/action: <...> | continuity: <anchor>. "
        "Do not add extra commentary or numbering outside the specified format."
    )
    return system, user


def parse_shotlist_lines(lines: List[str], max_len_name: int = 20) -> List[Tuple[str, str]]:
    """Parse shot list lines into (shot_key, tokens) pairs.

    - shot_key becomes an ordinal + short-name slug (underscored) when possible.
    - tokens is a comma-separated amalgam of the useful fields (excluding duration).
    """
    shots: List[Tuple[str, str]] = []
    idx = 1
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        parts = [p.strip() for p in text.split("|")]
        # Determine short name
        short = None
        if parts:
            head = parts[0]
            if ":" in head:
                short = head.split(":", 1)[1].strip()
        short_slug = slugify(short or f"shot-{idx}", max_len=max_len_name, add_hash=False).replace("-", "_")
        shot_key = f"{idx:02d}_{short_slug}"

        # Compose tokens: drop duration field; keep others
        keep_secs: List[str] = []
        for sec in parts:
            if sec.lower().startswith("duration"):
                continue
            if ":" in sec and sec.lower().split(":", 1)[0].strip() in {"shot", "lens", "camera", "subject/action", "continuity"}:
                # remove the field label
                keep_secs.append(sec.split(":", 1)[1].strip())
            else:
                # generic section
                keep_secs.append(sec)
        tokens = ", ".join(s for s in keep_secs if s)
        shots.append((shot_key, tokens))
        idx += 1
    return shots


def _normalize_for_match(token: str) -> str:
    t = token.strip().lower()
    for ch in "()[]{}":
        t = t.replace(ch, "")
    t = " ".join(t.split())
    return t


def _split_tokens(text: str) -> List[str]:
    return [p.strip() for p in text.split(",")]


def compose_seed(
    scenario: str,
    shot_tokens: str,
    bundle_tokens: List[str],
) -> str:
    """Compose a comma-separated seed string from scenario + shot + bundles."""
    parts: List[str] = []
    # Allow scenario in any language, but keep output style requirements English-only via system prompt.
    # Encourage concise English hints alongside scenario.
    if scenario.strip():
        parts.append(scenario.strip())
    if shot_tokens.strip():
        parts.append(shot_tokens.strip())
    for b in bundle_tokens:
        if b.strip():
            parts.append(b.strip())
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scenario-based Qwen-Image prompt maker (cuts + bundles)")
    ap.add_argument("--scenario", default=None, help="Scenario/story text (short). Use --scenario-file to read from file.")
    ap.add_argument("--scenario-file", default=None, help="File containing scenario/story text")
    ap.add_argument("--preset", default="storyboard", help="Shot preset name (default: storyboard)")
    ap.add_argument(
        "--bundle",
        action="append",
        default=[],
        choices=list(default_bundles().keys()),
        help="Add predefined bundle(s): quality-outfit, quality-character, nsfw-soft, nsfw-boudoir",
    )
    ap.add_argument(
        "--extra-bundle",
        action="append",
        default=[],
        help="Extra comma-separated tokens to append (repeatable)",
    )
    ap.add_argument("--variants", type=int, default=3, help="Variants per shot (default: 3)")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    ap.add_argument("--style", dest="qwen_style", choices=["sentence", "structured", "tags"], default="sentence", help="Qwen-Image output style")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q5_K_M", help="Ollama model name")
    ap.add_argument("--llm-host", default="http://localhost:11434", help="Ollama host base URL")
    ap.add_argument("--out-dir", default="output/scenarios", help="Base output directory (default: output/scenarios)")
    ap.add_argument("--slug-max-len", type=int, default=80, help="Max length for scenario folder slug (default: 80, hash suffix added if truncated)")
    ap.add_argument("--name", default=None, help="Scenario name for folder slug (default: derived from scenario)")
    ap.add_argument("--system-prompt-file", default="QWEN Image Creation Prompt Engineer Guide.txt", help="System prompt file to enforce guidelines")
    ap.add_argument("--chat", dest="llm_mode", action="store_const", const="chat", default="generate", help="Use Ollama /api/chat instead of /api/generate")
    ap.add_argument("--debug", action="store_true", help="Print raw/normalized outputs per line")
    ap.add_argument("--no-safe-adult-tags", dest="safe_adult_tags", action="store_false", help="Disable rewriting ambiguous age tags")
    ap.set_defaults(safe_adult_tags=True)

    # Adult-only controls
    ap.add_argument("--adult-only", action="store_true", help="Force adult-only hints in seed (adds 'adult woman')")
    ap.add_argument("--adult-flag-filenames", action="store_true", help="Append '_adult' to shot filenames and mark index")
    ap.add_argument("--adult-reject-minor", action="store_true", help="Filter out outputs containing minor-coded terms")
    ap.add_argument("--adult-banned", action="append", default=[], help="Additional banned minor-coded tokens (repeatable, comma-separated allowed)")
    ap.add_argument("--adult-banned-file", default=None, help="File with additional banned tokens (comma/newline separated)")

    # Story generation from topics/keywords
    ap.add_argument("--topic", action="append", default=[], help="Topic/keyword(s) to include in the scenario (repeatable)")
    ap.add_argument("--topic-file", default=None, help="File with topics (comma/newline separated)")
    ap.add_argument("--auto-scenario", action="store_true", help="Generate scenario from provided topics via LLM")
    ap.add_argument("--story-sentences", type=int, default=2, help="Number of sentences for auto scenario (default: 2)")
    ap.add_argument("--story-language", choices=["en", "ko"], default="ko", help="Language of the auto scenario (default: ko)")
    ap.add_argument("--story-style", choices=["logline", "vignette"], default="logline", help="Scenario style for auto generation")

    # Sequence/shot-list options (10s micro-video style)
    ap.add_argument("--num-cuts", type=int, default=None, help="Number of cuts in the sequence (e.g., 3..10)")
    ap.add_argument("--duration-sec", type=int, default=10, help="Approx total duration in seconds (default: 10)")
    ap.add_argument("--sequence-auto", action="store_true", help="Auto-generate a coherent shot list from the scenario via LLM")

    args = ap.parse_args()

    # Scenario text (or auto-generate from topics)
    scenario_text = args.scenario
    if args.scenario_file and (not scenario_text):
        scenario_text = read_text_if_exists(Path(args.scenario_file))

    # Collect topics
    topics: List[str] = []
    topics.extend(args.topic or [])
    if args.topic_file:
        t = read_text_if_exists(Path(args.topic_file))
        if t:
            for part in t.replace("\n", ",").split(","):
                p = part.strip()
                if p:
                    topics.append(p)

    # Auto-generate scenario if requested or if no scenario but topics present
    if (args.auto_scenario or not scenario_text) and topics:
        sys_prompt, user_prompt = build_story_prompt(
            topics=topics,
            sentences=int(args.story_sentences),
            language=args.story_language,
            style=args.story_style,
        )
        try:
            # Prefer chat; fallback to generate with ChatML
            scenario_text = ollama_chat(
                host=args.llm_host,
                model=args.model,
                system=sys_prompt,
                user=user_prompt,
                temperature=float(args.temperature),
            ).strip()
            if not scenario_text:
                chatml = build_qwen_chatml(sys_prompt, user_prompt)
                scenario_text = ollama_generate(
                    host=args.llm_host,
                    model=args.model,
                    prompt=chatml,
                    temperature=float(args.temperature),
                ).strip()
        except Exception as e:
            raise SystemExit(f"Failed to auto-generate scenario from topics: {e}")

    if not scenario_text:
        raise SystemExit("Provide --scenario/--scenario-file or use --topic/--auto-scenario")

    # System prompt override (QWEN guide by default)
    system_prompt_override = read_text_if_exists(Path(args.system_prompt_file)) or None

    # Bundles
    bundles_map = default_bundles()
    selected_bundle_tokens: List[str] = []
    for key in args.bundle or []:
        selected_bundle_tokens.extend(bundles_map.get(key, []))
    for extra in args.extra_bundle or []:
        # allow comma-separated additions
        for part in extra.replace("\n", ",").split(","):
            token = part.strip()
            if token:
                selected_bundle_tokens.append(token)
    # Adult-only seed hint
    if args.adult_only:
        selected_bundle_tokens.append("adult woman")

    # Shots: either auto-generated sequence or preset-based (optionally trimmed/extended)
    shots: List[Tuple[str, str]]
    if args.sequence_auto and (args.num_cuts and args.num_cuts > 0):
        sys_prompt, user_prompt = build_shotlist_prompt(
            scenario=scenario_text,
            topics=topics,
            num_cuts=int(args.num_cuts),
            duration_sec=int(args.duration_sec),
            language=args.story_language,
        )
        try:
            if args.llm_mode == "chat":
                raw = ollama_chat(
                    host=args.llm_host,
                    model=args.model,
                    system=sys_prompt,
                    user=user_prompt,
                    temperature=float(args.temperature),
                )
            else:
                # try generate then ChatML
                raw = ollama_generate(
                    host=args.llm_host,
                    model=args.model,
                    prompt=user_prompt,
                    temperature=float(args.temperature),
                )
                if not raw:
                    chatml = build_qwen_chatml(sys_prompt, user_prompt)
                    raw = ollama_generate(
                        host=args.llm_host,
                        model=args.model,
                        prompt=chatml,
                        temperature=float(args.temperature),
                    )
        except Exception as e:
            raise SystemExit(f"Failed to auto-generate shot list: {e}")
        lines = [ln for ln in (raw or "").splitlines() if ln.strip()]
        shots = parse_shotlist_lines(lines)[: int(args.num_cuts)]
        if not shots:
            shots = shot_preset(args.preset)
            if args.num_cuts:
                shots = shots[: int(args.num_cuts)]
    else:
        shots = shot_preset(args.preset)
        if args.num_cuts:
            # trim or extend by cycling presets
            base = shots
            need = int(args.num_cuts)
            if need <= len(base):
                shots = base[:need]
            else:
                extra: List[Tuple[str, str]] = []
                idx = len(base) + 1
                i = 0
                while len(base) + len(extra) < need:
                    key, tok = base[i % len(base)]
                    extra.append((f"{idx:02d}_seq", tok))
                    idx += 1
                    i += 1
                shots = base + extra

    # Output dir
    scenario_name = args.name or scenario_text
    slug = slugify(scenario_name, max_len=int(args.slug_max_len), add_hash=True)
    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare adult-ban set
    banned_norm = {
        "loli",
        "teen",
        "teenage",
        "schoolgirl",
        "underage",
        "minor",
        "child",
        "kid",
        "young girl",
        "young boy",
    }
    # extend banned set from CLI
    for raw in (args.adult_banned or []):
        for part in raw.replace("\n", ",").split(","):
            p = part.strip()
            if p:
                banned_norm.add(_normalize_for_match(p))
    if args.adult_banned_file:
        t = read_text_if_exists(Path(args.adult_banned_file))
        if t:
            for part in t.replace("\n", ",").split(","):
                p = part.strip()
                if p:
                    banned_norm.add(_normalize_for_match(p))

    # Generate per shot
    for shot_key, shot_tokens in shots:
        seed = compose_seed(scenario_text, shot_tokens, selected_bundle_tokens)
        seed = sanitize_seed(seed, safe_adult_tags=bool(args.safe_adult_tags))

        suffix = "_adult" if args.adult_flag_filenames else ""
        file_path = out_dir / f"{shot_key}{suffix}.txt"
        produced = 0
        filtered = 0
        with file_path.open("w", encoding="utf-8") as fh:
            for line in iter_qwen_image_prompts(
                seeds=[seed],
                host=args.llm_host,
                model=args.model,
                variants_per_seed=max(1, int(args.variants)),
                temperature=float(args.temperature),
                style=args.qwen_style,
                mode=args.llm_mode,
                debug=bool(args.debug),
                qwen_chatml_fallback=True,
                system_prompt_override=system_prompt_override,
                safe_adult_tags=bool(args.safe_adult_tags),
            ):
                if args.adult_reject_minor:
                    tokens = _split_tokens(line)
                    if any(_normalize_for_match(t) in banned_norm for t in tokens):
                        filtered += 1
                        continue
                fh.write(line.rstrip("\n") + "\n")
                produced += 1
        if filtered:
            print(f"[OK] {file_path} ({produced} line(s), filtered {filtered})")
        else:
            print(f"[OK] {file_path} ({produced} line(s))")

    # Scenario summary + small index file
    index_path = out_dir / "INDEX.txt"
    with index_path.open("w", encoding="utf-8") as idx:
        idx.write(f"Scenario: {scenario_name}\n")
        if topics:
            idx.write(f"Topics: {', '.join(topics)}\n")
        idx.write("\n--- Scenario Text ---\n")
        idx.write(scenario_text.strip() + "\n")
        idx.write("\n--- Settings ---\n")
        idx.write(f"Model: {args.model}\n")
        idx.write(f"Style: {args.qwen_style}\n")
        idx.write(f"Preset: {args.preset}\n")
        if args.num_cuts:
            idx.write(f"Num cuts: {int(args.num_cuts)}\n")
        if args.duration_sec:
            idx.write(f"Duration: ~{int(args.duration_sec)}s\n")
        if args.adult_only or args.adult_flag_filenames or args.adult_reject_minor:
            idx.write("Adult options:\n")
            if args.adult_only:
                idx.write("- adult_only (added 'adult woman' to seed)\n")
            if args.adult_flag_filenames:
                idx.write("- adult_flag_filenames (files suffixed with _adult)\n")
            if args.adult_reject_minor:
                idx.write("- adult_reject_minor (filtered minor-coded tokens)\n")
        if args.bundle:
            idx.write(f"Bundles: {', '.join(args.bundle)}\n")
        if args.extra_bundle:
            idx.write(f"Extra: {', '.join(args.extra_bundle)}\n")
        idx.write("Files:\n")
        for shot_key, _ in shots:
            idx.write(f"- {shot_key}.txt\n")
    print(f"[INDEX] {index_path}")


if __name__ == "__main__":
    main()
