# 프롬프트 모음 생성기 (Python)

이 저장소는 이미지 생성 모델용 프롬프트를 정리·확장하기 위한 간단한 Python 도구를 제공합니다. 기본 긍정/부정 프롬프트 파일을 생성하고, 원하면 로컬 LLM(Ollama)을 통해 유사한 긍정 프롬프트를 추가 생성할 수 있습니다.

## 도구별 주요 기능

### generate_prompts
- 디렉토리/파일 스캐폴딩: `output/positive.txt`, `output/negative.txt` 생성
- 긍정 프롬프트: 라인당 1개 프롬프트 저장(와일드카드 친화)
- LLM 통합: Ollama로 유사 긍정 프롬프트 변형 생성(variants)
- Qwen-Image 모드: 가이드 템플릿(sentence/structured/tags) 출력 지원
- 안전 기본값: 모호한 연령 표현을 성인 표현으로 치환(옵션 해제 가능)
- 제어: 제외 토큰(drop/reject), 시스템 프롬프트 파일/프리셋, ChatML 폴백 등

### scenario_prompt_maker
- 시나리오 입력 → 컷(샷) 프리셋별 프롬프트 생성
- Quality 번들(의상/인물) 및 사용자 정의 토큰 추가
- 토픽 기반 시나리오 자동 생성(`--auto-scenario`)
- 시퀀스 모드: 컷 수/길이에 맞는 샷리스트 자동 설계(`--sequence-auto`)
- 안전한 폴더명: 슬러그 길이 제한 + 해시 접미사
- 산출물 인덱스: `INDEX.txt`에 시나리오/설정/파일 목록 기록

## 요구 사항
- Python 3.8+
- LLM 생성을 사용하려면 Ollama 설치 및 모델 준비

### GPU 가속 관련(Windows/Linux)
- 본 도구는 Python 스크립트가 Ollama API를 호출하는 구조로, 가속은 Ollama가 담당합니다.
- NVIDIA GPU 사용 시 권장 사항
  - 최신 NVIDIA 그래픽 드라이버 설치 후 `nvidia-smi`가 정상 동작해야 합니다.
  - Ollama는 기본적으로 GPU를 자동 사용합니다(CUDA Toolkit 별도 설치는 일반적으로 필요 없음).
  - VRAM 가이드(대략):
    - 7–8B(q5 계열): 6–8GB VRAM 권장
    - 14B(q4 계열): 12–16GB VRAM 권장
    - 30B(q4 계열): 24–32GB VRAM 권장
  - GPU가 인식되지 않을 때 CPU로 강제 전환: `OLLAMA_NO_GPU=1` 환경변수 설정
- AMD GPU(Linux, ROCm): 배포판/드라이버 환경에 따라 지원이 제한적일 수 있습니다(실험적). `rocminfo`로 확인하세요.

## 운영체제별 실행 환경
아래는 Python과 Ollama(선택)의 설치 및 확인 방법입니다.

### Ubuntu (Linux)
- Python 설치(권장):
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```
- Ollama 설치(선택, LLM 사용 시):
```bash
curl -fsSL https://ollama.com/install.sh | sh
# 서비스 확인
ollama --version && ollama list
```
- NVIDIA GPU 드라이버(선택, 가속 시 권장):
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
# 재부팅 후
nvidia-smi
```
- 모델 준비(예):
```bash
ollama pull qwen2.5:7b-instruct-q5_K_M
# 동작 확인
ollama run qwen2.5:7b-instruct-q5_K_M "hello"
```
- 실행:
```bash
python3 scripts/generate_prompts.py --skip-base --llm \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --seed "cat eye shape, almond eyes, sharp eyeliner, ..." \
  --variants 5 --append
```

### macOS (Apple Silicon 포함)
- Python: 기본 내장되지만 최신 버전을 권장합니다(Homebrew 예시).
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3
```
- Ollama(선택):
```bash
brew install ollama
ollama --version && ollama list
```
- 모델 준비(예):
```bash
ollama pull llama3.1:8b-instruct-q5_K_M
ollama pull qwen2.5:7b-instruct-q5_K_M
```
- 실행(멀티라인 예시):
```bash
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --from-file output/positive.txt \
  --variants 3 \
  --append
```

### Windows 11/10
- Python 설치: https://www.python.org/downloads/ 에서 설치(또는 Microsoft Store).
  - PowerShell에서 버전 확인: `python --version` 또는 `py -3 --version`
- Ollama(선택): Windows용 설치 프로그램 사용(https://ollama.com/). 설치 후 PowerShell에서:
```powershell
ollama --version; ollama list
ollama pull qwen2.5:7b-instruct-q5_K_M
```
- NVIDIA GPU 가속(선택): 최신 NVIDIA 드라이버 설치 후 시스템 재부팅 → PowerShell에서 다음 실행
```powershell
nvidia-smi
```
- GPU 자동 사용이 어려우면 CPU 강제 전환(일시):
```powershell
$env:OLLAMA_NO_GPU = "1"; ollama run qwen2.5:7b-instruct-q5_K_M "hello"
```
- 실행:
  - 한 줄 실행(권장):
```powershell
python scripts/generate_prompts.py --skip-base --llm --model "qwen2.5:7b-instruct-q5_K_M" --variants 3 --append
```
  - PowerShell 멀티라인은 백틱(`) 사용:
```powershell
python scripts/generate_prompts.py `
  --skip-base --llm `
  --model "qwen2.5:7b-instruct-q5_K_M" `
  --variants 3 `
  --append
```
  - CMD.exe에서는 캐럿(^) 사용:
```cmd
python scripts\generate_prompts.py ^
  --skip-base --llm ^
  --model "qwen2.5:7b-instruct-q5_K_M" ^
  --variants 3 ^
  --append
```

## generate_prompts 사용법

## 빠른 시작
```bash
# 기본 긍정/부정 프롬프트 파일 생성
python3 scripts/generate_prompts.py

# 결과
# - output/positive.txt: 샘플 긍정 프롬프트 1줄
# - output/negative.txt: 샘플 부정 프롬프트 1줄
```

생성된 긍정 프롬프트는 라인당 1개의 콤마 구분 토큰 목록으로 구성되어, 와일드카드 확장에 바로 활용할 수 있습니다.

## LLM으로 유사 프롬프트 생성 (Ollama)
1) Ollama 설치(택1)
- 홈페이지에서 설치 패키지 다운로드 후 설치(맥OS 권장)
- 또는 Homebrew: `brew install ollama`

2) 모델 준비(예시)
```bash
# 권장 예시 모델(속도/품질 밸런스)
ollama pull llama3.1:8b-instruct-q5_K_M
ollama pull qwen2.5:7b-instruct-q5_K_M
```

3) 스크립트 실행 예시
```bash
# 기존 positive.txt에 유사 프롬프트 5줄 추가
python3 scripts/generate_prompts.py --skip-base --llm --variants 5 --append

# 특정 모델과 시드를 직접 지정
python3 scripts/generate_prompts.py \
  --llm \
  --model "llama3.1:8b-instruct-q5_K_M" \
  --seed "cat eye shape, almond eyes, sharp eyeliner, ..." \
  --variants 3 \
  --append

# 파일에서 여러 시드를 읽어 각각 변형 생성
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --from-file output/positive.txt \
  --variants 3 \
  --append
```

기본 LLM 호스트는 `http://localhost:11434` 이며, Ollama가 백그라운드에서 실행 중이어야 합니다.

### CLI 옵션 요약 (generate_prompts)
```text
--out-dir <PATH>            출력 디렉토리 (기본: output)
--append                    기존 파일에 이어쓰기
--llm                       LLM을 사용해 유사 긍정 프롬프트 생성
--model <NAME>              Ollama 모델명 (기본: llama3.1:8b-instruct-q5_K_M)
--llm-host <URL>            Ollama 호스트 URL (기본: http://localhost:11434)
--variants <N>              시드당 생성할 변형 개수 (기본: 3)
--temperature <F>           샘플링 온도 (기본: 0.7)
--seed <TEXT>               단일 시드 문자열 (우선순위 높음)
--from-file <FILE>          시드 목록 파일(라인당 1개)
--llm-mode <generate|chat>  Ollama API 모드 선택 (기본: generate)
--debug-llm                 원본/정규화 출력 디버그 로그
--skip-base                 기본 샘플 파일 쓰기 생략
--variants-out <FILE>       변형 결과를 별도 파일에 기록
--incremental               생성 즉시 1라인씩 곧바로 append
--fsync                     각 라인 쓰기 후 fsync(안전, 느림)
--progress-every <N>        N라인마다 진행률 출력(0=끄기)
--system-prompt <TEXT>      chat/ChatML에서 사용할 시스템 프롬프트
--system-prompt-file <FILE> 파일에서 시스템 프롬프트 읽기(텍스트보다 우선)
--system-prompt-preset <NAME>
                             내장 프리셋 사용(텍스트/파일 미지정 시 적용)
--no-qwen-chatml-fallback   Qwen용 ChatML 폴백 비활성화
--qwen-image                 Qwen-Image 공식 가이드라인 기반 프롬프트 생성
--qwen-style <sentence|structured|tags>
                             출력 형식 선택(기본: sentence)
--qwen-out <FILE>            Qwen-Image 결과를 별도 파일에 기록
--exclude <TOKENS>          제외 토큰(콤마 구분), 반복 지정 가능
--exclude-file <FILE>       제외 토큰 파일(라인 또는 콤마 구분)
--exclude-mode <drop|reject>제외 방식: drop=토큰만 제거, reject=라인 폐기
--retries <N>               reject 모드에서 각 라인 재시도 횟수(기본 3)
--no-safe-adult-tags        모호한 연령 표현 치환 비활성화
```

 

## 기존 텍스트 파일로부터 생성
- 보유 중인 텍스트 파일(라인당 1개 시드 프롬프트)을 그대로 입력으로 사용할 수 있습니다.
- 각 라인은 “콤마로 구분된 긍정 태그” 형식이 권장됩니다.

예시:
```bash
# my_seeds.txt(라인당 1개)로부터 시드 읽고, 변형을 별도 파일로 저장
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --from-file my_seeds.txt \
  --variants 5 \
  --variants-out output/variants.txt \
  --incremental --progress-every 50 --append
```

### 제외 키워드 활용
원치 않는 태그가 포함되는 것을 방지하려면 제외 토큰을 지정하세요.

```bash
# 1) 토큰만 제거(drop)
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --from-file my_seeds.txt \
  --exclude "animal ears, hat" \
  --exclude-file exclude.txt \
  --exclude-mode drop \
  --variants 10 --variants-out output/variants.txt --incremental --append

# 2) 해당 토큰이 포함된 라인을 아예 버리고 재시도(reject)
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --seed "cat eye shape, almond eyes, sharp eyeliner, ..." \
  --exclude "hat" \
  --exclude-mode reject --retries 5 \
  --variants 100 --incremental --progress-every 10 --append
```
토큰 매칭은 대소문자 무시, 괄호/가중치 제거 뒤 비교합니다. 예) `((hat))`도 `hat`과 일치로 간주합니다.

### 시스템 프롬프트를 파일로 지정
긴 시스템 프롬프트는 파일로 관리하는 것이 편리합니다. `--system-prompt-file`이 지정되면 `--system-prompt`보다 우선합니다.

```bash
# 예: system_prompt.txt 파일을 사용
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --from-file my_seeds.txt \
  --system-prompt-file system_prompt.txt \
  --variants 20 --incremental --progress-every 10 --append
```

### 시스템 프롬프트 프리셋
파일/텍스트 대신 내장 프리셋을 사용할 수 있습니다. 현재 지원: `illustrious-xl`.

```bash
# Illustrious-XL 프리셋(태그형 변형 생성)
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --from-file my_seeds.txt \
  --system-prompt-preset illustrious-xl \
  --variants 10 --variants-out output/variants.txt --append

# Illustrious-XL 프리셋(Qwen-Image 모드)
python3 scripts/generate_prompts.py \
  --skip-base --llm --qwen-image \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --from-file my_seeds.txt \
  --qwen-style sentence \
  --system-prompt-preset illustrious-xl \
  --variants 5 --qwen-out output/qwen_image_prompts.txt --append
```
우선순위: `--system-prompt-file` > `--system-prompt` > `--system-prompt-preset` > 내장 기본값

## 시나리오 프롬프트 제작기(컷/샷)
`scripts/scenario_prompt_maker.py`는 “상황/이야기(Scenario)”를 입력으로 받아, 스토리보드 컷(샷) 프리셋에 따라 Qwen-Image 가이드 형식으로 프롬프트를 생성합니다. 옵션으로 Quality(이미지 품질 지향) 의상/인물 키워드 번들을 추가할 수 있습니다.

- 시나리오 기반: `--scenario` 또는 `--scenario-file`
- 컷 프리셋: `--preset storyboard`(기본) – Establishing, Wide, Medium, Close-up, Over-Shoulder, Detail
- 스타일: `--style sentence|structured|tags` (Qwen-Image 가이드)
- 번들(선택): `--bundle quality-outfit`, `--bundle quality-character` (추가 토큰은 `--extra-bundle`)
 - 번들(선택): `--bundle quality-outfit`, `--bundle quality-character`, `--bundle nsfw-soft`, `--bundle nsfw-boudoir` (추가 토큰은 `--extra-bundle`)
- 시스템 프롬프트: 기본값으로 `QWEN Image Creation Prompt Engineer Guide.txt`를 사용
- 결과: `output/scenarios/<slug>/<NN>_<shot>.txt` 파일들로 저장

### 시퀀스 모드(10초 영상 컷 플래닝)
- 컷 수 지정: `--num-cuts 3`(최소) ~ `--num-cuts 10`(권장)
- 길이 목표: `--duration-sec 10`(기본)
- 자동 샷리스트 생성: `--sequence-auto` 사용 시, 시나리오/토픽을 바탕으로 일관된 시퀀스(연속성, 카메라 움직임, 앵글)를 LLM이 설계하고, 컷별 프롬프트를 생성합니다.
- 토픽으로 시나리오도 함께 자동 생성하려면 `--topic ... --auto-scenario`를 병행하세요.

예시(영문 시나리오 자동 생성 + 5컷 시퀀스 + 태그형 출력):
```bash
python3 scripts/scenario_prompt_maker.py \
  --topic "proposal in a fancy restaurant" --topic "rainstorm" \
  --auto-scenario --story-language en --story-sentences 3 \
  --sequence-auto --num-cuts 5 --duration-sec 10 \
  --style tags --variants 1
```

예시(태그형, 번들 2종, 컷별 3변형):
```bash
python3 scripts/scenario_prompt_maker.py \
  --scenario "Rainy neon alley; a poised woman exits a jazz bar" \
  --bundle quality-outfit --bundle quality-character \
  --style tags --variants 3 \
  --model "qwen2.5:7b-instruct-q5_K_M"
```

예시(라벨형, 시스템 프롬프트 기본값 사용):
```bash
python3 scripts/scenario_prompt_maker.py \
  --scenario "Dawn on a foggy harbor, minimal elegance" \
  --style structured --variants 2
```

출력 구조 예시:
```text
output/scenarios/
  rainy-neon-alley-a-poised-woman-exits-a-jazz-bar/
    01_establishing.txt
    02_wide.txt
    03_medium.txt
    04_closeup.txt
    05_over_shoulder.txt
    06_detail.txt
    INDEX.txt
```

팁
- 시나리오는 한국어여도 무방하나, 시스템 규칙상 출력은 영어로 1줄 형식을 엄격히 따릅니다.
- `--extra-bundle "comma, separated, tokens"`로 자유 키워드를 추가할 수 있습니다.

### CLI 옵션 요약 (scenario_prompt_maker)
```text
--scenario <TEXT>                    시나리오(짧은 문장). 파일 입력은 --scenario-file
--scenario-file <FILE>               시나리오 텍스트 파일
--topic <TEXT>                       토픽/키워드(반복 지정). 파일 입력은 --topic-file
--topic-file <FILE>                  토픽 파일(콤마/개행 구분)
--auto-scenario                      토픽 기반 시나리오 자동 생성 활성화
--story-sentences <N>                자동 시나리오 문장 수(기본 2)
--story-language <en|ko>             자동 시나리오 언어(기본 ko)
--story-style <logline|vignette>     자동 시나리오 스타일(기본 logline)

--preset <NAME>                      컷 프리셋(기본 storyboard)
--num-cuts <N>                       컷 수 지정(예: 3–10)
--duration-sec <N>                   전체 길이 목표(초, 기본 10)
--sequence-auto                      시나리오로부터 샷리스트 자동 설계

--bundle <quality-outfit|quality-character|nsfw-soft|nsfw-boudoir>
                                     품질/NSFW(비노골적) 번들(반복 가능)
--extra-bundle <TOKENS>              추가 토큰(콤마 구분, 반복 가능)

--style <sentence|structured|tags>   Qwen-Image 출력 형식(기본 sentence)
--variants <N>                       컷별 변형 개수(기본 3)
--temperature <F>                    샘플링 온도(기본 0.7)
--model <NAME>                       Ollama 모델명(기본 qwen2.5:7b-instruct-q5_K_M)
--llm-host <URL>                     Ollama 호스트(기본 http://localhost:11434)
--llm-mode <generate|chat>           API 모드(기본 generate, --chat 별칭 지원)
--system-prompt-file <FILE>          시스템 프롬프트 파일(기본 QWEN 가이드 파일)
--no-safe-adult-tags                 모호한 연령 표현 치환 비활성화

--out-dir <DIR>                      출력 베이스(기본 output/scenarios)
--slug-max-len <N>                   시나리오 폴더 슬러그 최대 길이(기본 80, 해시 접미사)
--name <TEXT>                        폴더명 슬러그에 쓸 표시 이름
--debug                              LLM 원본/정규화 출력 디버그
 
성인 전용 옵션
--adult-only                         시드에 'adult woman' 힌트를 추가하여 성인만 대상으로 강제
--adult-flag-filenames               생성 파일명을 *_adult.txt 형태로 저장하고 INDEX에 표시
--adult-reject-minor                 미성년 관련 금칙어 포함 라인 필터링(파일 기록 안 함)
--adult-banned <TOKENS>              금칙어 추가(반복 가능, 콤마 구분)
--adult-banned-file <FILE>           금칙어 추가 파일(콤마/개행 구분)
```


### Qwen-Image 프롬프트 생성 모드 (generate_prompts)
`--qwen-image`를 사용하면 시드 토큰(`--seed` / `--from-file`)을 바탕으로 Qwen-Image 가이드라인에 맞춘 프롬프트를 생성합니다.

- 스타일 선택(`--qwen-style`):
  - `sentence`: 1줄 내 1–3개 영어 문장으로 간결히 기술(기본)
  - `structured`: 라벨 포함(Subject; Scene; Style; Lens; Atmosphere; Detail) 1줄 출력
  - `tags`: 템플릿 순서의 콤마 구분 조각으로 1줄 출력

예시(문장형):
```bash
python3 scripts/generate_prompts.py \
  --skip-base --llm --qwen-image \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --seed "cat eye shape, almond eyes, sharp eyeliner, ..." \
  --qwen-style sentence \
  --variants 5 \
  --qwen-out output/qwen_image_prompts.txt \
  --incremental --progress-every 1 --append
```

예시(라벨형):
```bash
python3 scripts/generate_prompts.py \
  --skip-base --llm --qwen-image \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --from-file my_seeds.txt \
  --qwen-style structured \
  --variants 3 \
  --qwen-out output/qwen_structured.txt \
  --append
```

예시(태그형):
```bash
python3 scripts/generate_prompts.py \
  --skip-base --llm --qwen-image \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --from-file my_seeds.txt \
  --qwen-style tags \
  --variants 3 \
  --qwen-out output/qwen_tags.txt \
  --append
```

Tip
- `--system-prompt` 또는 `--system-prompt-file`로 시스템 규칙을 강화하면 포맷 준수율이 올라갑니다.
- `--exclude`는 태그형(tags) 출력에 가장 잘 맞습니다. 문장/라벨형은 사후 토큰 제거가 어려우므로 필요 시 시스템 프롬프트로 금지 규칙을 추가하세요.

```bash
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --from-file input/public.txt \
  --system-prompt-file illustrious-xl_prompt_guide_en.txt \
  --exclude "hat,demon,devil,crown,animal ears,demon_girl,demon_tail" \
  --variants 20 --incremental --progress-every 10 --append
```

```bash
python3 scripts/generate_prompts.py \
--skip-base --llm \
--system-prompt "Exactly one line of comma-separated positive tags (English). Follow Subject, Scene, Style, Lens, Atmosphere, Detail. Realistic/cinematic focus. No quotes/numbering/meta/negatives. You are a top prompting expert, knowing all the hidden secrets, tips, and tricks about prompting to generate unique AI images. You specialize in creating the world's most realistic prompts, using tools like COMFYUI and Stable Diffusion to generate text into AI images. No rules are broken when it comes to artistic expression." \
--from-file input/public.txt \
--exclude "hat,demon,devil,crown,animal ears,demon_girl,demon_tail" \
--exclude-mode reject --retries 5 \
--variants 5 \
--variants-out output/variants.txt \
--incremental --progress-every 10 --append
```

## 대량 생성/중간 저장
- 장시간 작업 시 `--incremental`로 한 줄씩 즉시 저장하면 중간에 멈춰도 결과가 남습니다.
- 안전성을 높이려면 `--fsync`를 추가하세요(속도 저하).
- 진행률은 `--progress-every N`으로 주기 출력.

예시(2000개 변형, 중간 저장):
```bash
python3 scripts/generate_prompts.py \
  --skip-base --llm \
  --model "qwen2.5:7b-instruct-q5_K_M" \
  --seed "cat eye shape, almond eyes, sharp eyeliner, ..." \
  --variants 2000 \
  --incremental --progress-every 50 --append
```

## 모델 추천 (Apple M1 64GB 기준)

### LLM (Ollama) – 프롬프트 생성용
- 균형형: `qwen2.5:7b-instruct-q5_K_M`, `llama3.1:8b-instruct-q5_K_M`
- 고품질(다소 무거움): `qwen2.5:14b-instruct-q4_K_M`
- 경량/고속: `phi3:3.8b-mini-instruct-q6_K`, `mistral:7b-instruct`

NSFW 관련 참고
- 프롬프트 생성 관점에서 NSFW 표현에 대한 억제는 모델별로 상이합니다. 일반적으로 Qwen 2.5 계열이 Llama3.1보다 완곡하며, Mistral 7B Instruct도 상대적으로 관대한 편입니다. 다만 어느 경우든 미성년 관련 내용은 절대 금지하며, 현지 법과 플랫폼 정책을 준수하세요.

### 이미지 모델 (외부 파이프라인 예시)
- 범용: Stable Diffusion XL 1.0(SDXL), Stable Diffusion 1.5 + 커스텀 체크포인트/LoRA
- 리얼리즘 계열(예시): Realistic Vision, Deliberate
- 애니/일러스트 계열(예시): Anything v5, AbyssOrangeMix3

NSFW 관련 참고
- 위 일부 커스텀 체크포인트는 NSFW 생성에 관대할 수 있습니다. 사용 국가/서비스의 정책을 준수하고, 성인만을 명시적으로 대상으로 하며, 착취적/불법 콘텐츠는 절대 금지하세요. 본 저장소는 시각적 성인/미성년 모호성을 줄이도록 기본적으로 모호한 태그를 성인 표현으로 치환합니다.

> 참고: Qwen3 계열은 Ollama 패키징/프롬프트 템플릿이 제각각이라 한 줄 출력 준수성이 낮을 수 있습니다. 본 도구는 “정확히 1줄” 출력을 기대하므로, 우선 Llama3.1 · Qwen2.5를 권장합니다.

---

## 💻 로컬 실행 가능한 Roleplay / NSFW LLM 목록

| 구분 | 모델명                                               | 파라미터             | 주요 특징                                                |
| -- | ------------------------------------------------- | ---------------- | ---------------------------------------------------- |
| 1  | **Blue-Orchid-2x7B**                              | 2×7B (MoE)       | Explicit RP용 MoE 모델 (Dialogue + Storywriting 전문가 분리) |
| 2  | **Mistral 22B**                                   | 22B              | 검열 적음, 캐릭터 일관성 보통, 큰 VRAM 요구                         |
| 3  | **L3.1 Euryale 2.2**                              | 70B+             | 고품질 Roleplay, 대형 서버급 VRAM 필요                         |
| 4  | **Midnight Miqu 103B**                            | 103B             | 몰입감 강함, 64GB 이상 RAM 권장                               |
| 5  | **Magnum 123B / 70B**                             | 70~123B          | 대형 고품질 NSFW 모델                                       |
| 6  | **Luminum 123B**                                  | 123B             | Magnum 계열, 창의적 RP에 강함                                |
| 7  | **Wizard2 8×22B**                                 | 176B (8×22B MoE) | 거대 MoE 구조, 고성능                                       |
| 8  | **Stheno 3.2**                                    | 13B 정도           | RTX 3070급에서도 구동 가능                                   |
| 9  | **Gemmasutra**                                    | 약 13B            | 언센서드, 감정 표현 우수                                       |
| 10 | **Dirty-Muse-Writer-v01-Uncensored-Erotica-NSFW** | 13B              | NSFW 전문 튜닝 모델                                        |
| 11 | **Llama-3.2-uncensored-erotica / unsloth.F16**    | 8–13B            | Llama 3.2 기반 언센서드 버전                                 |
| 12 | **Llama-3.1-405B-Instruct (Q4–Q8 quant)**         | 405B (양자화 버전)    | 초대형 오프라인 모델 (LM Studio, Koboldcpp 지원)                |
| 13 | **NousResearch / Hermes-3-Llama-3.1-405B**        | 405B             | Roleplay 품질 우수, Llama3 기반                            |
| 14 | **Goliath 120B**                                  | 120B             | 고성능 구형 대형 모델                                         |
| 15 | **Mistral Small**                                 | 7B               | “may generate offensive material”, 낮은 검열             |
| 16 | **Euryel / Euryale (구버전)**                        | 30–70B           | Euryale 초기 버전, 일부 QLoRA 양자화 존재                       |

> 안전 메모: NSFW/Roleplay 사용 시 항상 성인을 전제로 하고, 현지 법/플랫폼 정책을 준수하세요. 미성년, 착취, 비동의 콘텐츠는 절대 금지입니다.

---

## ⚙️ 실행 환경 요약

| GPU / RAM                               | 실행 가능한 모델                                                       |
| --------------------------------------- | --------------------------------------------------------------- |
| **RTX 3060~3070 (8–12 GB)**             | Stheno 3.2, Mistral 7B Small, Blue-Orchid-2x7B (Q4), Gemmasutra |
| **RTX 3090 / 4080 / 4090 (24 GB)**      | Mistral 22B (Q4), Dirty-Muse, Llama 3.2 Unsloth F16             |
| **서버급 (A6000, 3090×2, 64 GB RAM 이상)**   | Euryale 2.2, Midnight Miqu 103B, Magnum 123B                    |
| **128 GB RAM + CPU inference (no GPU)** | Llama 3.1 405B Q4_K_M via LM Studio or koboldcpp                |

---

## 📦 다운로드 위치 (Hugging Face)

| 모델명                                   | Hugging Face Repo                                                                                   |
| ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Blue-Orchid-2x7B                      | [nakodanei/Blue-Orchid-2x7b](https://huggingface.co/nakodanei/Blue-Orchid-2x7b)                     |
| Llama-3.1-405B-Instruct               | [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)     |
| Hermes-3-Llama-3.1-405B               | [NousResearch/Hermes-3-Llama-3.1-405B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B) |
| Dirty-Muse-Writer-v01                 | [Dirty-Muse-Writer-v01-Uncensored-Erotica-NSFW](https://huggingface.co/models) (검색 필요)              |
| Llama-3.2-uncensored-erotica          | 검색어: `Llama-3.2-uncensored-erotica unsloth`                                                         |
| Stheno 3.2 / Gemmasutra / Magnum 123B | [huggingface.co/models](https://huggingface.co/models)에서 직접 검색                                      |


## 파일 구조
```text
output/
  positive.txt   # 라인당 1개 긍정 프롬프트
  negative.txt   # 기본 샘플 1라인
scripts/
  generate_prompts.py
```

## 동작 개요
- `scripts/generate_prompts.py` 실행 시 기본 샘플 긍정/부정 프롬프트를 생성합니다.
- `--llm` 사용 시 시드(직접 입력/파일/기존 positive.txt)로부터 유사 긍정 프롬프트를 생성하여 `positive.txt` 끝에 추가합니다.
- 안전 기본값으로 `1girl`/`girl` 등 모호한 표현은 `1woman`/`woman`으로 치환됩니다. 원문 유지가 필요하면 `--no-safe-adult-tags`를 사용하세요.

## 주의 및 팁
- Ollama 모델은 사전 다운로드가 필요합니다(`ollama pull <모델명>`).
- LLM 출력이 여러 줄/설명형으로 나오는 경우가 드물게 있습니다. 스크립트가 1줄만 취하도록 후처리하지만, 필요하면 프롬프트 템플릿을 더 엄격하게 조정할 수 있습니다.
- Qwen(non-instruct) 모델은 `generate` 모드에서 빈 응답이 나올 수 있어 ChatML 폴백을 자동 시도합니다. 필요 시 `--llm-mode chat` 또는 `--no-qwen-chatml-fallback`를 사용하세요.
- 중복 줄이 생기면 간단히 정리할 수 있습니다:
  - `awk '!seen[$0]++' output/positive.txt > output/positive.tmp && mv output/positive.tmp output/positive.txt`


## 라이선스
내부/연구용으로 자유롭게 사용하세요. 별도 라이선스가 필요하다면 알려주세요.

## ComfyUI 테크니컬 가이드

본 도구가 생성한 프롬프트를 ComfyUI로 전달·자동화하는 다양한 실전 방법을 정리했습니다. ComfyUI는 한 줄 강제가 아니므로, 태그 + 시나리오 문장(렌즈/카메라/조명)을 적절히 섞은 멀티라인도 사용할 수 있습니다.

기본 워크플로 개요(SDXL 예)
- Checkpoint Loader (SDXL) → 모델/CLIP/VAE 로드
- CLIP Text Encode (SDXL) ×2 → 긍정/부정 프롬프트 인코딩
- Empty Latent Image → 해상도 지정(예: 1024×1024)
- KSampler (SDXL) → sampler/steps/cfg/seed 제어
- VAE Decode → Save Image

컷(시퀀스) 운용 팁(3–10컷)
- 수동: 컷 파일별로 프롬프트 교체 후 순차 실행
- 배치: 커뮤니티 노드(ComfyUI-Manager로 ‘text’/‘loop’ 검색)로 파일→반복 실행 구성

일관성(Continuity)
- 공통 시드/카메라/렌즈/조명을 컷 전반에 유지
- IP-Adapter로 스타일 고정, ControlNet(Depth/Lineart/SoftEdge)로 구도 유지
- SDXL Refiner(denoise 0.2–0.4), ESRGAN 업스케일로 품질 보강

—

# 1) ComfyUI HTTP API로 “템플릿 워크플로우” 재전송 (가장 단순/확실)

ComfyUI의 `/prompt` 큐는 상태를 기억하지 않아서 매 실행마다 그래프(JSON)를 보내는 게 정석입니다. 템플릿 workflow.json을 저장해두고, 그 안의 CLIPTextEncode(양의 프롬프트) 노드의 `text`만 교체해서 POST 하면 됩니다.

### 1-A. 템플릿 준비

- ComfyUI에서 현재 워크플로우 Export → `workflow.template.json` 저장
- 포지티브 프롬프트 위치(보통 `CLIPTextEncode`의 `inputs.text`)에 토큰 넣기: `"text": "__POS__"`

### 1-B. 실행용 JSON 만들고 큐에 넣기 (sed + curl)

```bash
# 프롬프트 문자열만 바꿔서 보낼 JSON 생성
POS="cinematic portrait, korean actress, soft light, 50mm"
sed "s|__POS__|${POS//|/\|}|g" workflow.template.json > payload.json

# ComfyUI 프롬프트 큐에 전송 (기본 포트 8188)
curl -s -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  -d @payload.json | jq .
```

팁: ComfyUI는 `client_id` 사용을 권장합니다. 템플릿 루트에 `"client_id": "innofree-cli"` 같은 필드를 추가하면 트래킹이 편합니다.

### 1-C. Python(단일 스크립트)로 교체·전송

```python
#!/usr/bin/env python3
import json, requests

tmpl = json.load(open("workflow.template.json"))
pos = "cinematic portrait, korean actress, soft light, 50mm"

def replace(obj):
    if isinstance(obj, dict):
        for k,v in obj.items():
            if k == "text" and isinstance(v, str) and v == "__POS__":
                obj[k] = pos
            else:
                replace(v)
    elif isinstance(obj, list):
        for i in obj:
            replace(i)

replace(tmpl)
tmpl.setdefault("client_id","innofree-cli")
r = requests.post("http://127.0.0.1:8188/prompt", json=tmpl, timeout=60)
print(r.status_code, r.text[:300])
```

—

# 2) “파일 로더 노드” + 파일 덮어쓰기 (그래프는 고정, 문자열만 교체)

그래프를 매번 안 보내려면 텍스트 파일을 읽는 노드(예: `Load Text From File`)를 CLIPTextEncode 앞에 두고, 실행 전마다 파일만 갱신합니다.

흐름
1. 워크플로우: `Text From File` → `CLIPTextEncode`
2. 프롬프트 파일 경로: `/data/prompts/positive.txt`
3. 실행 전 교체:

```bash
cat > /data/prompts/positive.txt <<'EOF'
cinematic portrait, korean actress, soft light, 50mm
EOF
# ComfyUI UI에서 Run or Queue 트리거 (또는 API로 트리거 노드 호출)
```

장단점
- 장점: 대용량 그래프를 매번 안 보내도 됨
- 단점: 해당 커스텀 노드 설치 필요, 트리거는 별도

—

# 3) MCP(Model Context Protocol) 브릿지로 “프롬프트-주입 툴” 만들기 (LLM·에이전트 연결)

MCP 서버(파이썬/노드)에서 “ComfyUI로 이미지 생성” 툴을 노출하고, 내부적으로 템플릿 JSON의 `__POS__` 치환 → `/prompt` POST를 수행합니다. ChatGPT/에디터 MCP 클라이언트에서 도구 호출만으로 ComfyUI 파이프라인을 돌릴 수 있습니다.

### 3-A. Node(TypeScript) 미니 서버 예시

```ts
// mcp-comfy.ts
import { Server } from "@modelcontextprotocol/sdk/server/mcp";
import axios from "axios";
import fs from "fs";

const server = new Server({
  name: "comfyui-bridge",
  version: "0.1.0",
  tools: [
    {
      name: "comfy_generate",
      description: "Send a positive prompt to ComfyUI using a workflow template.",
      inputSchema: {
        type: "object",
        properties: { positive: { type: "string" } },
        required: ["positive"],
      },
      handler: async ({ positive }) => {
        const tmpl = JSON.parse(fs.readFileSync("workflow.template.json","utf8"));
        const replace = (o:any) => {
          if (Array.isArray(o)) o.forEach(replace);
          else if (o && typeof o === "object") {
            for (const k of Object.keys(o)) {
              if (k === "text" && o[k] === "__POS__") o[k] = positive;
              else replace(o[k]);
            }
          }
        };
        replace(tmpl);
        tmpl.client_id = tmpl.client_id ?? "innofree-mcp";
        const { data } = await axios.post("http://127.0.0.1:8188/prompt", tmpl, { timeout: 60000 });
        return { content: [{ type: "text", text: JSON.stringify(data).slice(0,500) }] };
      },
    },
  ],
});

server.start();
```

설치 개요

```bash
npm i @modelcontextprotocol/sdk axios
ts-node mcp-comfy.ts # 또는 빌드 후 node 실행
```

MCP 클라이언트(에디터·LLM)가 이 서버를 등록하면, `comfy_generate` 툴 호출 시 `positive`만 넘겨 ComfyUI를 실행합니다.

### 3-B. Python MCP 서버도 유사하게 가능
- `modelcontextprotocol` 파이썬 SDK 사용
- 로직 동일: 템플릿 로드 → `__POS__` 치환 → `/prompt` POST

—

## 어떤 방법을 쓰면 좋을까?

- 가장 간단/안전: 1) 템플릿 JSON 재전송
- 그래프 고정, 문자열만 바꾸기: 2) 파일 로더 노드
- LLM/에이전트와 일원화: 3) MCP 브릿지

## 기타

- 네거티브 프롬프트: `__NEG__` 토큰을 네거티브쪽 CLIPTextEncode에 두고 동일 방식으로 교체
- Client-ID/히스토리: `client_id` 고정 시 `/history/{prompt_id}` 조회·로깅 용이
- 모델/로라/샘플러/스텝: 템플릿에 `__CKPT__`, `__LORA__`, `__SAMPLER__`, `__STEPS__` 토큰을 추가해 일괄 치환
