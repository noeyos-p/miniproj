"""
통합 Vision Assistant Backend
- mini1: 실시간 객체 인식 및 거리 측정
- mini2: 이미지/비디오 질문-응답 (Qwen2-VL)
"""

import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
# Mac 전용 환경변수

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# mini1과 mini2의 모든 변수와 함수를 가져오기 위한 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'mini1'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mini2'))

# mini1 앱 임포트 (전체 앱을 서브앱으로 마운트)
from mini1.app import app as mini1_app, pipeline

# mini2 앱 임포트
from mini2.main import app as mini2_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행"""
    print("[메인] 통합 서버 시작")

    # mini2 모델 로드
    print("[메인] mini2 모델 로드 중...")
    from mini2.main import load_models
    load_models()
    print("[메인] mini2 모델 로드 완료")

    yield

    # 종료 시 정리
    print("[메인] 서버 종료 중...")

    # mini1 파이프라인 정지
    pipeline.running = False

    # mini2 모델 정리
    import gc
    import torch
    from mini2 import main as mini2_main
    if hasattr(mini2_main, 'vl_model'):
        del mini2_main.vl_model, mini2_main.vl_processor
        del mini2_main.translator, mini2_main.translator_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[메인] 정리 완료")

# 메인 앱 생성
app = FastAPI(
    title="Vision Assistant API",
    description="시각장애인을 위한 통합 비전 어시스턴트",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "Vision Assistant API",
        "services": {
            "mini1": "실시간 객체 인식 및 거리 측정 (/mini1)",
            "mini2": "이미지/비디오 질문-응답 (/api)"
        }
    }

# 헬스체크
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# mini1 앱을 /mini1 경로에 마운트
app.mount("/mini1", mini1_app)

# mini2 앱을 루트에 마운트 (이미 /api 프리픽스 사용 중)
app.mount("/", mini2_app)

if __name__ == "__main__":
    import uvicorn

    # 템플릿 디렉토리가 없으면 생성 (mini1 전용)
    templates_dir = os.path.join(os.path.dirname(__file__), 'mini1', 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # access_log=False로 반복되는 /process_frame 로그 숨김
    # 서버 시작 등 중요한 로그는 표시됨
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
