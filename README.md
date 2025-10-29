# 프롬프트 모음 생성기 (Python)

이 저장소는 이미지 생성 모델용 프롬프트를 정리·확장하기 위한 간단한 Python 도구를 제공합니다. 기본 긍정/부정 프롬프트 파일을 생성하고, 원하면 로컬 LLM(Ollama)을 통해 유사한 긍정 프롬프트를 추가 생성할 수 있습니다.

## 주요 기능
- 디렉토리/파일 스캐폴딩: `output/positive.txt`, `output/negative.txt` 생성
- 긍정 프롬프트: 와일드카드용으로 라인당 1개 프롬프트 저장
- 부정 프롬프트: 기본 샘플 1라인 저장(원하면 파일로 관리 가능)
- LLM 통합(옵션): 로컬 Ollama 서버로 유사 프롬프트 변형 생성 및 추가
- 안전 기본값: 모호한 연령 표현(`1girl`, `girl`)을 성인 표현(`1woman`, `woman`)으로 치환(옵션으로 해제 가능)

## 요구 사항
- Python 3.8+
- (선택) LLM 생성을 사용하려면 macOS Apple Silicon(M1/M2/M3) 등에서 Ollama 설치 및 모델 준비

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

## CLI 옵션 요약
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
--no-qwen-chatml-fallback   Qwen용 ChatML 폴백 비활성화
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
- 균형형: `llama3.1:8b-instruct-q5_K_M`, `qwen2.5:7b-instruct-q5_K_M`
- 고품질(다소 무거움): `qwen2.5:14b-instruct-q4_K_M`
- 경량/고속: `phi3:3.8b-mini-instruct-q6_K`

> 참고: Qwen3 계열은 Ollama 패키징/프롬프트 템플릿이 아직 제각각이라 한 줄 출력 준수성이 낮을 수 있습니다. 본 도구는 “정확히 1줄” 출력을 기대하므로, 우선 Llama3.1 · Qwen2.5를 권장합니다.

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
