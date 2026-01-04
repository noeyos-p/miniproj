import { useState, useRef, useCallback, useEffect } from 'react';
import './ObjectDetection.css';

// 프로덕션: 빈 문자열(상대 경로), 개발: localhost:8000
// const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '');
const API_URL = import.meta.env.VITE_API_URL || '';
const DOUBLE_CLICK_DELAY = 400;

const FEATURE_GUIDE = `
시각장애인 보조 서비스 사용 가능한 기능을 안내합니다.

첫째, 화면을 두 번 터치하면 시스템을 시작하거나 정지할 수 있습니다.

둘째, 하단의 마이크 버튼을 두 번 터치하면 음성 명령을 녹음할 수 있습니다.
시작해, 정지해, 말해줘, 조용히해, 종료해 등의 명령어를 사용할 수 있습니다.

셋째, 음소거 버튼을 두 번 터치하면 현재 재생 중인 음성 안내를 중지합니다.

넷째, 말하기 버튼을 두 번 터치하면 현재 화면의 상태를 음성으로 안내합니다.

다섯째, 종료 버튼을 두 번 터치하면 시스템을 종료합니다.

모든 버튼은 한 번 터치하면 해당 버튼의 설명을, 두 번 터치하면 기능을 실행합니다.
`;

interface Detection {
  box: [number, number, number, number];
  label: string;
  distance: number;
  roi: [number, number];
  track_id?: number;
  confidence?: number;
}

interface FrameSize {
  width: number;
  height: number;
}

function ObjectDetection() {
  // 상태
  const [isRunning, setIsRunning] = useState(false);
  const [statusText, setStatusText] = useState('화면을 두 번 터치하면 시작합니다.');
  const [statusColor, setStatusColor] = useState('white');
  const [isRecording, setIsRecording] = useState(false);

  // Refs
  const cameraPreviewRef = useRef<HTMLVideoElement>(null);
  const cameraCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const cameraStreamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const isProcessingRef = useRef(false);

  const clickTimerRef = useRef<number | null>(null);
  const screenClickTimerRef = useRef<number | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);

  // TTS 함수
  const speak = useCallback((text: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
  }, []);

  // 오버레이에 감지 결과 그리기 (다중 객체 지원)
  const drawDetection = useCallback((detections: Detection[] | null, frameSize: FrameSize) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) {
      console.log('[Draw Debug] Canvas ref가 없음');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.log('[Draw Debug] Context를 가져올 수 없음');
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    console.log('[Draw Debug] detections:', detections);
    console.log('[Draw Debug] frameSize:', frameSize);

    if (!detections || detections.length === 0) {
      console.log('[Draw Debug] 감지된 객체 없음');
      return;
    }

    console.log('[Draw Debug] 객체 그리기 시작 - 총', detections.length, '개');

    const scaleX = canvas.width / frameSize.width;
    const scaleY = canvas.height / frameSize.height;

    // ROI 가이드 라인 (첫 번째 객체의 ROI 사용)
    const [roiLeft, roiRight] = detections[0].roi;
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(roiLeft * scaleX, 0);
    ctx.lineTo(roiLeft * scaleX, canvas.height);
    ctx.moveTo(roiRight * scaleX, 0);
    ctx.lineTo(roiRight * scaleX, canvas.height);
    ctx.stroke();

    // 모든 감지된 객체 그리기
    detections.forEach((detection) => {
      const { box, label, distance, track_id } = detection;
      const [x1, y1, x2, y2] = box;

      // 감지 박스
      ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

      // 레이블 (Track ID 포함)
      ctx.fillStyle = 'rgba(255, 0, 0, 1)';
      ctx.font = '20px Arial';
      const labelText = track_id !== undefined && track_id !== null
        ? `ID:${track_id} ${label} ${distance}m`
        : `${label} ${distance}m`;
      ctx.fillText(labelText, x1 * scaleX, (y1 - 10) * scaleY);
    });
  }, []);

  // 프레임 캡처 및 전송
  const captureAndSendFrame = useCallback(async () => {
    if (!cameraStreamRef.current) return;
    if (isProcessingRef.current) return;

    const video = cameraPreviewRef.current;
    const canvas = cameraCanvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    isProcessingRef.current = true;

    try {
      ctx.drawImage(video, 0, 0);

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.7);
      });

      if (!blob) return;

      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');

      // mini1 API 경로로 수정
      const response = await fetch(`${API_URL}/mini1/process_frame`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        // 디버깅: 받은 데이터 확인
        console.log('[Frontend Debug] 받은 데이터:', data);
        console.log('[Frontend Debug] detection 타입:', typeof data.detection, Array.isArray(data.detection));
        console.log('[Frontend Debug] detection 내용:', data.detection);

        drawDetection(data.detection, data.frame_size);

        if (data.speech && data.speech.length > 0) {
          speak(data.speech);
        }
      }
    } catch (error) {
      console.error('프레임 전송 오류:', error);
    } finally {
      isProcessingRef.current = false;
    }
  }, [drawDetection, speak]);

  // 카메라 시작
  const startCamera = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });

      cameraStreamRef.current = stream;

      if (cameraPreviewRef.current) {
        cameraPreviewRef.current.srcObject = stream;
      }

      return true;
    } catch (error) {
      console.error('카메라 접근 오류:', error);
      setStatusText('카메라 권한을 허용해주세요');
      return false;
    }
  }, []);

  // 카메라 중지
  const stopCamera = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (cameraPreviewRef.current) {
      cameraPreviewRef.current.srcObject = null;
    }
  }, []);

  // 비디오 메타데이터 로드 시 캔버스 크기 설정 및 프레임 캡처 시작
  const handleVideoLoaded = useCallback(() => {
    const video = cameraPreviewRef.current;
    const canvas = cameraCanvasRef.current;
    const overlay = overlayCanvasRef.current;

    if (video && canvas && overlay) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;

      // 프레임 캡처 시작 (200ms = 5fps)
      frameIntervalRef.current = window.setInterval(captureAndSendFrame, 200);
    }
  }, [captureAndSendFrame]);

  // 파이프라인 토글
  const togglePipeline = useCallback(async () => {
    const endpoint = isRunning ? `${API_URL}/mini1/stop` : `${API_URL}/mini1/start`;

    try {
      const response = await fetch(endpoint, { method: 'POST' });
      const data = await response.json();

      if (data.status === 'started' || data.status === 'already_running') {
        setIsRunning(true);
        setStatusText('작동 중... 두 번 터치하면 정지합니다.');
        setStatusColor('#4CAF50');

        const cameraStarted = await startCamera();
        if (!cameraStarted) return;
      } else {
        setIsRunning(false);
        setStatusText('정지됨. 두 번 터치하면 시작합니다.');
        setStatusColor('#FF5252');
        stopCamera();
      }
    } catch (error) {
      console.error('파이프라인 토글 오류:', error);
      speak('연결 오류가 발생했습니다.');
    }
  }, [isRunning, startCamera, stopCamera, speak]);

  // 화면 클릭 처리
  const handleScreenClick = useCallback(() => {
    if (screenClickTimerRef.current) {
      clearTimeout(screenClickTimerRef.current);
      screenClickTimerRef.current = null;
      window.speechSynthesis.cancel();
      togglePipeline();
    } else {
      screenClickTimerRef.current = window.setTimeout(() => {
        if (isRunning) {
          speak('현재 작동 중입니다. 두 번 클릭하면 정지합니다.');
        } else {
          speak('현재 정지 상태입니다. 두 번 클릭하면 시작합니다.');
        }
        screenClickTimerRef.current = null;
      }, DOUBLE_CLICK_DELAY);
    }
  }, [isRunning, togglePipeline, speak]);

  // 버튼 클릭 처리 (한 번: 설명, 두 번: 기능 실행)
  const handleButtonClick = useCallback(
    (e: React.MouseEvent, description: string, action?: () => void) => {
      e.stopPropagation();

      if (clickTimerRef.current) {
        clearTimeout(clickTimerRef.current);
        clickTimerRef.current = null;
        window.speechSynthesis.cancel();
        action?.();
      } else {
        clickTimerRef.current = window.setTimeout(() => {
          speak(description);
          clickTimerRef.current = null;
        }, DOUBLE_CLICK_DELAY);
      }
    },
    [speak]
  );

  // 명령 전송
  const sendCommand = useCallback(
    async (commandText: string) => {
      try {
        const response = await fetch(`${API_URL}/mini1/command`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: commandText }),
        });
        const data = await response.json();

        if (data.response) {
          speak(data.response);
        }
      } catch (error) {
        console.error('명령 전송 오류:', error);
      }
    },
    [speak]
  );

  // 마이크 초기화
  const initMicrophone = useCallback(async (): Promise<boolean> => {
    if (audioStreamRef.current?.active) return true;

    try {
      audioStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      return true;
    } catch (error) {
      console.error('마이크 접근 오류:', error);
      setStatusText('마이크 권한을 허용해주세요');
      return false;
    }
  }, []);

  // 녹음 시작
  const startRecording = useCallback(async () => {
    if (!audioStreamRef.current?.active) {
      setStatusText('마이크 권한이 필요합니다. 페이지를 새로고침하세요.');
      return;
    }

    try {
      const mediaRecorder = new MediaRecorder(audioStreamRef.current);
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        try {
          setStatusText('음성 인식 중...');
          const response = await fetch(`${API_URL}/mini1/voice`, {
            method: 'POST',
            body: formData,
          });
          const data = await response.json();
          setStatusText('인식: ' + data.text);

          if (data.response) {
            speak(data.response);
          }
        } catch (error) {
          console.error('음성 전송 오류:', error);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('녹음 시작 오류:', error);
    }
  }, [speak]);

  // 녹음 중지
  const stopRecording = useCallback(() => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') return;

    mediaRecorderRef.current.stop();
    setIsRecording(false);
  }, []);

  // 녹음 토글
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // 페이지 로드 시 초기화
  useEffect(() => {
    initMicrophone();

    const timer = setTimeout(() => {
      speak('시각장애인 보조 서비스입니다. 버튼을 한 번 터치하면 설명을, 두 번 터치하면 기능을 실행합니다.');
    }, 1000);

    return () => {
      clearTimeout(timer);
      stopCamera();
    };
  }, [initMicrophone, speak, stopCamera]);

  return (
    <div className="blind-app" onClick={handleScreenClick}>
      <div className="status-text" style={{ color: statusColor }}>
        {statusText}
      </div>

      {/* 정보 버튼 */}
      <button
        className="info-button"
        onClick={(e) =>
          handleButtonClick(
            e,
            '정보 버튼입니다. 두 번 클릭하면 사용 가능한 모든 기능을 안내합니다.',
            () => speak(FEATURE_GUIDE)
          )
        }
      >
        ℹ️
      </button>

      {/* 비디오 컨테이너 */}
      <div className="video-container" style={{ display: isRunning ? 'flex' : 'none' }}>
        <video
          ref={cameraPreviewRef}
          className="camera-preview"
          autoPlay
          playsInline
          muted
          onLoadedMetadata={handleVideoLoaded}
          style={{ display: isRunning ? 'block' : 'none' }}
        />
        <canvas
          ref={overlayCanvasRef}
          className="overlay-canvas"
          style={{ display: isRunning ? 'block' : 'none' }}
        />
        <canvas ref={cameraCanvasRef} style={{ display: 'none' }} />
      </div>

      {/* 컨트롤 패널 */}
      <div className="control-panel">
        <button
          className={`control-btn record-btn ${isRecording ? 'recording' : ''}`}
          onClick={(e) =>
            handleButtonClick(
              e,
              '녹음 버튼입니다. 두 번 클릭하면 녹음을 시작하거나 중지합니다.',
              toggleRecording
            )
          }
        >
          {isRecording ? '🔴' : '🎤'}
        </button>

        <button
          className="control-btn"
          onClick={(e) =>
            handleButtonClick(e, '음소거 버튼입니다.', () => sendCommand('조용히해'))
          }
        >
          🔇
        </button>

        <button
          className="control-btn"
          onClick={(e) =>
            handleButtonClick(
              e,
              '말하기 버튼입니다. 현재 상태를 음성으로 안내합니다.',
              () => sendCommand('말해줘')
            )
          }
        >
          📢
        </button>

        <button
          className="control-btn exit-btn"
          onClick={(e) =>
            handleButtonClick(e, '종료 버튼입니다.', () => {
              sendCommand('종료해');
              setTimeout(() => window.location.href = '/', 1000);
            })
          }
        >
          🛑
        </button>
      </div>
    </div>
  );
}

export default ObjectDetection;
