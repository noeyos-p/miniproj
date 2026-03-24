# ICU

시각장애인을 위한 AI 보조 웹 애플리케이션. 실시간 객체 감지 및 거리 인식, 이미지/영상 기반 질의응답 기능을 음성 인터페이스와 함께 제공합니다.

---

## 주요 기능

- **실시간 객체 감지**: YOLOv8 기반 카메라 실시간 객체 탐지 및 거리 추정
- **이미지/영상 Q&A**: Qwen2-VL 기반 이미지 및 영상 분석 질의응답
- **음성 인터페이스**: STT(음성 인식) / TTS(음성 합성) 지원
- **다국어 지원**: M2M100 모델 기반 한영 자동 번역
- **스트리밍 응답**: SSE(Server-Sent Events) 기반 실시간 응답 스트리밍
- **접근성 최적화**: 더블탭 네비게이션, 고대비 UI, 음성 명령 지원

---

## 기술 스택

### Frontend
| 항목 | 기술 |
|------|------|
| Framework | React 19, TypeScript |
| Build Tool | Vite |
| Routing | React Router v7 |
| Speech | Web Speech API (STT/TTS) |
| Camera | MediaStream API |

### Backend
| 항목 | 기술 |
|------|------|
| Framework | FastAPI |
| Server | Uvicorn |
| Object Detection | YOLOv8 (Ultralytics) |
| Vision-Language | Qwen2-VL-2B-Instruct |
| Translation | M2M100-418M |
| Face Recognition | InsightFace |
| Pose Detection | MediaPipe |

### AI / ML
| 항목 | 기술 |
|------|------|
| Deep Learning | PyTorch (CUDA 12.6 / MPS / CPU) |
| Computer Vision | OpenCV, Pillow |
| NLP | Transformers (HuggingFace) |
| Speech | OpenAI Whisper |
| Inference | ONNX Runtime |

---

## 프로젝트 구조

```
ICU/
├── frontend/          # React + TypeScript 프론트엔드
└── backend/           # FastAPI 백엔드 (AI 서비스)
```

### Frontend (`frontend/src/`)
```
├── App.tsx                    # 메인 라우터
├── main.tsx                   # React 엔트리포인트
└── pages/
    ├── Home.tsx               # 서비스 선택 홈
    ├── ObjectDetection.tsx    # 실시간 객체 감지 UI
    └── VisionAssistant.tsx    # 이미지/영상 Q&A 인터페이스
```

### Backend (`backend/`)
```
├── main.py                    # Vision Q&A API 서버
├── requirements.txt           # Python 의존성
├── mini1/                     # 실시간 객체 감지 파이프라인
│   ├── app.py                 # 감지 웹 서버
│   ├── test_pipeline.py       # YOLO + 깊이 추정 핵심 로직
│   └── follow_up_service.py   # LLM 기반 객체 설명
└── mini2/                     # 대안 Vision 모델 구현
    └── main.py                # Qwen2-VL 구현
```

---

## 로컬 개발 환경 설정

### 사전 요구사항
- Python 3.10+
- Node.js 18+
- GPU 8GB+ VRAM 권장 (CPU 모드 가능)

### 1. Backend 실행

```bash
cd backend
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> AI 서버가 `http://localhost:8000` 에서 실행됩니다.
> 첫 실행 시 모델 다운로드 (~10GB+): Qwen2-VL, M2M100, YOLOv8

### 2. Frontend 실행

```bash
cd frontend
npm install
npm run dev
```

> 개발 서버가 `http://localhost:3000` 에서 실행됩니다.

---

## 환경 변수

### Backend (`.env`)
| 변수 | 설명 |
|------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 키 (후속 설명용) |
| `CUDA_VISIBLE_DEVICES` | GPU 선택 (선택사항) |

### Frontend
| 변수 | 설명 |
|------|------|
| `VITE_API_URL` | Backend API URL (기본: `http://localhost:8000`) |
