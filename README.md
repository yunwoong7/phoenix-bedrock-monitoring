<h2 align="center">
AI Research Assistant with Plan-Execute Pattern
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.12.7-blue.svg"/>
  <img src="https://img.shields.io/badge/boto3-v1.35.91-blue.svg"/>
  <img src="https://img.shields.io/badge/streamlit-v1.41.1-blue.svg"/>
</div>

이 프로젝트는 AI Agent의 구현 방법과 모니터링 설정을 테스트하기 위한 예제입니다. Plan-Execute 패턴을 활용하여 질문에 대한 계획 수립, 실행, 응답 생성을 수행하는 AI 연구 어시스턴트를 구현했습니다.

---

## 프로젝트 개요

이 프로젝트는 다음과 같은 핵심 기능을 구현합니다:
- LangGraph를 활용한 Plan-Execute 패턴의 AI Agent 구현
- 실시간 스트리밍을 통한 추론 과정 시각화
- Phoenix를 활용한 LLM 모니터링 및 관찰성 확보
- Streamlit 기반의 대화형 인터페이스

## 구현 상의 주요 도전과 해결 방법

### 1. LangGraph 스트리밍 처리
- **문제**: LangGraph의 노드 내부에서 실행되는 내용을 스트리밍으로 받아오기 어려움
- **해결**: `astream_events`를 활용하여 이벤트 기반의 스트리밍 구현
- **한계**: `on_chat_model_stream`에서 일부 정보 유실 발생

### 2. 네이밍 컨벤션 불일치 문제
- **문제**: LLM이 선호하는 camelCase와 프로젝트의 snake_case 사이의 불일치로 인한 데이터 유실
- **해결**: 프롬프트에 명시적으로 네이밍 규칙을 지정하여 해결

### 3. Plan 단계 스트리밍 제한
- **문제**: `with_structured_output` 사용 시 이전 내용이 포함된 청크 발생
- **해결**: `astream` 대신 `astream_events` 사용으로 우회

## 프로젝트 구조
```
src/
├── agent/
│   ├── nodes/           # Plan, Execute, Respond 노드 구현
│   ├── states/          # 상태 관리 스키마
│   └── workflow/        # LangGraph 워크플로우 정의
├── demo/                # Streamlit 애플리케이션
└── monitoring/          # Phoenix 모니터링 설정
```

`src/config/setting.yaml`에서 다음과 같이 각 단계별 모델을 설정할 수 있습니다.

## 설치 및 실행

```bash
# 환경 변수 설정
export TAVILY_API_KEY="your_key"
export PHOENIX_API_KEY="your_key"
export PHOENIX_PROJECT_NAME="your_project"
export PHOENIX_ENDPOINT="your_endpoint"

# 실행
streamlit run src/demo/app.py
```

## 주요 기술 스택
- LangGraph: AI Agent 워크플로우 구현
- Streamlit: 웹 인터페이스
- [Phoenix](https://app.phoenix.arize.com/): LLM 모니터링
- LangChain: LLM 통합 및 도구 연동
- Tavily: 웹 검색 기능

## 실행 화면 및 모니터링
### 애플리케이션 실행 화면
<div align="center">
<img src="https://github.com/user-attachments/assets/7a7d08cd-1537-40a0-b7e1-3b9c3dce6aba" width="70%">
</div>

### Phoenix 모니터링 대시보드
<div align="center">
<img src="https://github.com/user-attachments/assets/4d1b92d3-754c-4265-b41e-83c1c3344fc0" width="70%">
</div>