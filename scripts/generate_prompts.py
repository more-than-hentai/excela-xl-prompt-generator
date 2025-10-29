#!/usr/bin/env python3
"""
Prompt generator for positive/negative files, with optional LLM-based variants.

기능:
- 출력 디렉토리 구조 생성 (기본: output/)
- 긍정 프롬프트: wildcard 용도로 라인당 1프롬프트 저장 (positive.txt)
- 부정 프롬프트: 단일 라인 또는 라인별로 저장 가능 (negative.txt)
- LLM(로컬 Ollama 등) 기반 유사 프롬프트 생성 옵션

사용법 (기본 파일 생성):
  python3 scripts/generate_prompts.py
  python3 scripts/generate_prompts.py --out-dir custom       # 출력 디렉토리 지정
  python3 scripts/generate_prompts.py --append               # 기존 파일에 추가

사용법 (LLM으로 유사 프롬프트 생성):
  # 로컬 Ollama 서버(기본: http://localhost:11434) 사용
  python3 scripts/generate_prompts.py --llm --variants 5 \
      --model "llama3.1:8b-instruct-q5_K_M" \
      --from-file output/positive.txt --append

  # 단일 시드 문자열로 생성
  python3 scripts/generate_prompts.py --llm --seed "cat eye shape, ..." --variants 3

Ollama 참고:
- macOS Apple Silicon(M1/M2/M3)에서 로컬로 모델 구동 가능
- 권장 예시 모델: llama3.1 8B, qwen2.5 7B 등 (적절한 양자화)
- 설치/모델 준비 후: `ollama serve` (자동 실행), `ollama pull llama3.1:8b-instruct-q5_K_M`

주의(안전):
- 기본적으로 모호한 태그("1girl", "girl")는 성인 표현("1woman", "adult woman")으로 대체됩니다.
  원문 유지가 필요하면 `--no-safe-adult-tags` 옵션을 사용하세요.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


# 샘플 프롬프트 (요청 내용 기반)
SAMPLE_POSITIVES: list[str] = [
    (
        "cat eye shape, almond eyes, sharp eyeliner, foxy look, sharp jawline, "
        "elegant beauty, pale skin, glossy lips, eyelashes, (black eyes), long hair, "
        "(sexually_suggestive), looking at viewer, av_idol, k-illustration, "
        "(short torso, narrow_waist,naturally sagging breasts), , 1girl, ((sheer pantyhose)), "
        "micro tight skirt, breasts, solo, korean text, brown hair, long hair, jacket, bag, "
        "animal ears, cleavage, shirt, white shirt, black micro tight skirt, belt, handbag, "
        "pencil micro tight skirt, white background, fur trim, brown eyes, trembling, "
        "black jacket, simple background, blush, long sleeves, miniskirt, black legwear, "
        "large breasts, off shoulder, eyebrows visible through hair, brown legwear, "
        "closed mouth, bangs, medium breasts"
    )
]

SAMPLE_NEGATIVES: list[str] = [
    "(worst quality, low quality), (koam, 2d, anime, drawing, drawn face, cartoon, manga, cg, 3d, rendered), blush, ((hat)), closed eyes, pussy juice"
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: Iterable[str], append: bool = False) -> None:
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for line in lines:
            # 라인 종료 보장
            f.write(line.rstrip("\n") + "\n")


def normalize_prompt_line(text: str) -> str:
    # 한 줄, 앞뒤 공백/따옴표 제거, 연속 공백 정리
    line = text.strip().strip('"').strip("'")
    # Ollama가 코드블록을 반환하는 경우 제거
    if line.startswith("```") and line.endswith("```"):
        line = line.strip("`")
    # 여러 줄이라면 첫 줄만 사용
    if "\n" in line:
        line = line.splitlines()[0].strip()
    # 불필요한 끝 콤마 제거
    while line.endswith((",", ";")):
        line = line[:-1].rstrip()
    return line


def replace_ambiguous_age_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    banned = {"loli", "teen", "underage"}
    for t in tokens:
        key = t.strip()
        if not key:
            continue
        if key.lower() in banned:
            continue
        if key == "1girl":
            out.append("1woman")
        elif key.lower() == "girl":
            out.append("woman")
        else:
            out.append(key)
    return out


def _normalize_for_match(token: str) -> str:
    # Normalize token for comparison: lower, strip brackets/parentheses/weights, trim spaces
    t = token.strip().lower()
    # remove weighting braces/parentheses often used in prompts
    remove_chars = "()[]{}"
    for ch in remove_chars:
        t = t.replace(ch, "")
    # collapse whitespace
    t = " ".join(t.split())
    return t


def _split_tokens(text: str) -> List[str]:
    return [p.strip() for p in text.split(",")]


def _filter_excluded(tokens: Sequence[str], excluded_norm: Set[str]) -> List[str]:
    if not excluded_norm:
        return [t for t in tokens if t]
    kept: List[str] = []
    for t in tokens:
        if not t:
            continue
        if _normalize_for_match(t) in excluded_norm:
            continue
        kept.append(t)
    return kept


def sanitize_seed(seed: str, safe_adult_tags: bool = True, excluded_norm: Optional[Set[str]] = None) -> str:
    tokens = _split_tokens(seed)
    if safe_adult_tags:
        tokens = replace_ambiguous_age_tokens(tokens)
    if excluded_norm:
        tokens = _filter_excluded(tokens, excluded_norm)
    return ", ".join([t for t in tokens if t])


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    num_predict: int = 256,
    timeout: int = 60,
) -> str:
    """Call a local Ollama server's /api/generate endpoint and return text.

    Requires an Ollama server running at `host` and the given `model` pulled.
    """
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            text = obj.get("response", "")
            return text
    except HTTPError as e:
        # 가능한 경우 본문 메시지도 함께 노출
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        msg = f"Ollama HTTP {e.code}: {e.reason}"
        if body:
            msg += f" | {body}"
        raise RuntimeError(msg) from e
    except URLError as e:
        raise RuntimeError(
            "Failed to reach Ollama server. Is it running at "
            f"{host}? ({e.reason})"
        ) from e


def ollama_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    num_predict: int = 256,
    timeout: int = 60,
) -> str:
    """Call a local Ollama server's /api/chat endpoint and return text."""
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            # Ollama chat returns messages list or a single message
            if "message" in obj:
                content = obj["message"].get("content", "")
            elif "messages" in obj and obj["messages"]:
                content = obj["messages"][-1].get("content", "")
            else:
                content = obj.get("response", "")
            return content
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        msg = f"Ollama HTTP {e.code}: {e.reason}"
        if body:
            msg += f" | {body}"
        raise RuntimeError(msg) from e
    except URLError as e:
        raise RuntimeError(
            "Failed to reach Ollama server. Is it running at "
            f"{host}? ({e.reason})"
        ) from e


def build_variant_instruction(seed: str) -> str:
    return (
        "You are a prompt generator for composing POSITIVE tags for image models "
        "(e.g., Stable Diffusion).\n"
        "Output exactly ONE line: a comma-separated list of tokens, no quotes, "
        "no numbering, no extra lines. Keep the overall style, theme, and aesthetics "
        "similar to the given seed. Avoid illegal/harmful content. Do NOT output a "
        "negative prompt. If the seed contains '1girl' or 'girl', treat the subject as an "
        "adult woman.\n\n"
        f"Seed: {seed}\n"
        "One new similar positive prompt line:"
    )


def build_qwen_chatml(system: str, user: str) -> str:
    """Compose a ChatML-style prompt suitable for Qwen-style chat models.

    Many Qwen chat/base models respond better to explicit ChatML tokens when
    using the /api/generate endpoint instead of /api/chat.
    """
    return (
        "<|im_start|>system\n"
        + system.strip()
        + "\n<|im_end|>\n"
        + "<|im_start|>user\n"
        + user.strip()
        + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def generate_variants_with_ollama(
    seeds: Sequence[str],
    host: str,
    model: str,
    variants_per_seed: int,
    temperature: float,
    safe_adult_tags: bool,
    mode: str = "generate",
    debug: bool = False,
    qwen_chatml_fallback: bool = True,
    system_prompt_override: str | None = None,
    excluded_norm: Optional[Set[str]] = None,
    exclude_mode: str = "drop",
    retries: int = 3,
) -> List[str]:
    results: List[str] = []
    for line in iter_variants_with_ollama(
        seeds=seeds,
        host=host,
        model=model,
        variants_per_seed=variants_per_seed,
        temperature=temperature,
        safe_adult_tags=safe_adult_tags,
        mode=mode,
        debug=debug,
        qwen_chatml_fallback=qwen_chatml_fallback,
        system_prompt_override=system_prompt_override,
        excluded_norm=excluded_norm,
        exclude_mode=exclude_mode,
        retries=retries,
    ):
        results.append(line)
    return results


def iter_variants_with_ollama(
    seeds: Sequence[str],
    host: str,
    model: str,
    variants_per_seed: int,
    temperature: float,
    safe_adult_tags: bool,
    mode: str = "generate",
    debug: bool = False,
    qwen_chatml_fallback: bool = True,
    system_prompt_override: str | None = None,
    excluded_norm: Optional[Set[str]] = None,
    exclude_mode: str = "drop",
    retries: int = 3,
):
    """Yield one normalized variant string at a time (skips blanks),
    honoring exclusion rules and retries if configured.
    """
    excluded_norm = excluded_norm or set()

    for seed in seeds:
        clean_seed = sanitize_seed(seed, safe_adult_tags=safe_adult_tags, excluded_norm=excluded_norm)
        def attempt_once() -> str:
            instruction = build_variant_instruction(clean_seed)
            if mode == "chat":
                raw = ollama_chat(
                    host=host,
                    model=model,
                    system=(
                        system_prompt_override
                        if system_prompt_override is not None
                        else (
                            "You generate a single-line comma-separated list of positive tags "
                            "for image generation models. Output exactly one line, no quotes."
                        )
                    ),
                    user=instruction,
                    temperature=temperature,
                )
            else:
                raw = ollama_generate(host=host, model=model, prompt=instruction, temperature=temperature)
                if (not raw) and qwen_chatml_fallback and ("qwen" in model.lower()) and ("instruct" not in model.lower()):
                    sys_prompt = (
                        system_prompt_override
                        if system_prompt_override is not None
                        else (
                            "You are a prompt generator that outputs exactly one line: "
                            "a comma-separated list of positive tags for image generation."
                        )
                    )
                    chatml = build_qwen_chatml(sys_prompt, instruction)
                    raw = ollama_generate(host=host, model=model, prompt=chatml, temperature=temperature)
            line = normalize_prompt_line(raw)
            if debug:
                print(f"[LLM raw] {raw!r}\n[LLM norm] {line!r}")
            return line

        produced = 0
        for _ in range(variants_per_seed):
            # Try until accepted or retries exhausted
            got: Optional[str] = None
            for _try in range(max(1, retries + 1)):
                candidate = attempt_once()
                if not candidate:
                    continue
                tokens = _split_tokens(candidate)
                if not excluded_norm:
                    got = ", ".join([t for t in tokens if t])
                    break
                if exclude_mode == "drop":
                    kept = _filter_excluded(tokens, excluded_norm)
                    if kept:
                        got = ", ".join(kept)
                        break
                    else:
                        # all tokens excluded → treat as failure
                        continue
                else:  # reject
                    if any(_normalize_for_match(t) in excluded_norm for t in tokens):
                        # contains excluded → retry
                        continue
                    got = ", ".join([t for t in tokens if t])
                    break
            if got:
                produced += 1
                yield got


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate positive/negative prompt files.")
    parser.add_argument("--out-dir", default="output", help="Output directory (default: output)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing files instead of overwriting",
    )
    # LLM 옵션
    parser.add_argument("--llm", action="store_true", help="Use LLM to generate similar positive prompts")
    parser.add_argument("--model", default="llama3.1:8b-instruct-q5_K_M", help="LLM model name (Ollama)")
    parser.add_argument("--llm-host", default="http://localhost:11434", help="LLM host base URL (Ollama)")
    parser.add_argument("--variants", type=int, default=3, help="Variants per seed (LLM mode)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (LLM mode)")
    parser.add_argument("--seed", default=None, help="Single seed prompt string (overrides from-file)")
    parser.add_argument("--from-file", default=None, help="Read seeds from file (one per line)")
    parser.add_argument("--llm-mode", choices=["generate", "chat"], default="generate", help="Use /api/generate or /api/chat")
    parser.add_argument("--debug-llm", action="store_true", help="Print raw and normalized LLM outputs")
    parser.add_argument("--no-qwen-chatml-fallback", action="store_true", help="Disable ChatML fallback for Qwen models in generate mode")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt used for chat/ChatML modes")
    parser.add_argument("--system-prompt-file", default=None, help="Read system prompt from file (overrides --system-prompt)")
    parser.add_argument(
        "--no-safe-adult-tags",
        dest="safe_adult_tags",
        action="store_false",
        help="Do not rewrite ambiguous age tags (e.g., '1girl' -> '1woman')",
    )
    parser.set_defaults(safe_adult_tags=True)
    # 파일 제어 옵션
    parser.add_argument("--skip-base", action="store_true", help="Skip writing the built-in sample prompts")
    parser.add_argument("--variants-out", default=None, help="Write LLM variants to a separate file path")
    parser.add_argument("--incremental", action="store_true", help="Append each generated variant immediately")
    parser.add_argument("--fsync", action="store_true", help="fsync after each write (safer, slower)")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N appended lines (0=off)")
    # Exclusion control
    parser.add_argument("--exclude", action="append", default=[], help="Exclude tokens (comma-separated); may be repeated")
    parser.add_argument("--exclude-file", default=None, help="File with tokens to exclude (one per line or comma-separated)")
    parser.add_argument("--exclude-mode", choices=["drop", "reject"], default="drop", help="drop: remove tokens; reject: discard lines containing them")
    parser.add_argument("--retries", type=int, default=3, help="Retries per variant when exclude-mode=reject")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    positive_path = out_dir / "positive.txt"
    negative_path = out_dir / "negative.txt"

    if not args.skip_base:
        # 긍정: 라인당 1프롬프트 (wildcard 파일로 사용 가능)
        write_lines(positive_path, SAMPLE_POSITIVES, append=args.append)
        # 부정: 기본적으로 단일 라인 저장, 여러 라인도 가능
        write_lines(negative_path, SAMPLE_NEGATIVES, append=args.append)
        print(f"Wrote {positive_path} ({len(SAMPLE_POSITIVES)} lines)")
        print(f"Wrote {negative_path} ({len(SAMPLE_NEGATIVES)} line(s))")

    # LLM 기반 유사 프롬프트 생성 (옵션)
    if args.llm:
        # Resolve system prompt from file or flag
        system_prompt_val = args.system_prompt
        if args.system_prompt_file:
            try:
                system_prompt_val = Path(args.system_prompt_file).read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"[WARN] System prompt file not found: {args.system_prompt_file}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to read system prompt file: {e}", file=sys.stderr)
        # Prepare exclusion set
        excluded: Set[str] = set()
        def add_excluded_from_text(text: str):
            for part in text.replace("\n", ",").split(","):
                p = part.strip()
                if p:
                    excluded.add(_normalize_for_match(p))
        for raw in args.exclude:
            if raw:
                add_excluded_from_text(raw)
        if args.exclude_file:
            try:
                add_excluded_from_text(Path(args.exclude_file).read_text(encoding="utf-8"))
            except FileNotFoundError:
                print(f"[WARN] Exclude file not found: {args.exclude_file}", file=sys.stderr)

        # 시드 수집: --seed > --from-file > 기존 positive.txt > 샘플
        seeds: List[str]
        if args.seed:
            seeds = [args.seed]
        elif args.from_file:
            seeds = [s.strip() for s in Path(args.from_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        elif positive_path.exists():
            seeds = [s.strip() for s in positive_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        else:
            seeds = SAMPLE_POSITIVES[:]

        target_path = Path(args.variants_out) if args.variants_out else positive_path
        variants_per_seed = max(1, int(args.variants))

        if args.incremental:
            total_planned = len(seeds) * variants_per_seed
            appended = 0
            try:
                with target_path.open("a", encoding="utf-8") as out:
                    for line in iter_variants_with_ollama(
                        seeds=seeds,
                        host=args.llm_host,
                        model=args.model,
                        variants_per_seed=variants_per_seed,
                        temperature=float(args.temperature),
                        safe_adult_tags=bool(args.safe_adult_tags),
                        mode=args.llm_mode,
                        debug=bool(args.debug_llm),
                        qwen_chatml_fallback=not bool(args.no_qwen_chatml_fallback),
                        system_prompt_override=system_prompt_val,
                        excluded_norm=excluded,
                        exclude_mode=args.exclude_mode,
                        retries=int(args.retries),
                    ):
                        out.write(line + "\n")
                        out.flush()
                        if args.fsync:
                            try:
                                os.fsync(out.fileno())
                            except OSError:
                                pass
                        appended += 1
                        if args.progress_every and (appended % args.progress_every == 0):
                            print(f"[LLM] Appended {appended}/{total_planned} lines -> {target_path}")
            except KeyboardInterrupt:
                print(f"\n[LLM] Interrupted. Appended {appended}/{total_planned} lines to {target_path}")
                return
            except Exception as e:
                print(f"[LLM] Generation failed after {appended} lines: {e}", file=sys.stderr)
                sys.exit(2)
            print(f"[LLM] Completed. Appended {appended}/{total_planned} lines to {target_path}")
        else:
            try:
                variants = generate_variants_with_ollama(
                    seeds=seeds,
                    host=args.llm_host,
                    model=args.model,
                    variants_per_seed=variants_per_seed,
                    temperature=float(args.temperature),
                    safe_adult_tags=bool(args.safe_adult_tags),
                    mode=args.llm_mode,
                    debug=bool(args.debug_llm),
                    qwen_chatml_fallback=not bool(args.no_qwen_chatml_fallback),
                    system_prompt_override=system_prompt_val,
                    excluded_norm=excluded,
                    exclude_mode=args.exclude_mode,
                    retries=int(args.retries),
                )
            except Exception as e:
                print(f"[LLM] Generation failed: {e}", file=sys.stderr)
                sys.exit(2)

            write_lines(target_path, variants, append=True)
            print(f"[LLM] Appended {len(variants)} variant line(s) to {target_path}")


if __name__ == "__main__":
    main()
