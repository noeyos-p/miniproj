import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

function Home() {
  const navigate = useNavigate();
  const [isSpeaking, setIsSpeaking] = useState(false);
  const pendingActionRef = useRef<(() => void) | null>(null);

  // TTS
  const speak = useCallback((text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    utterance.rate = 0.9;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }, []);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  // 접근성: 버튼 선택 시 음성 안내, 더블클릭으로 실행
  const handleSelection = useCallback((label: string, action: () => void) => {
    return () => {
      speak(`${label} 선택됨. 실행하려면 화면을 두 번 두드리세요.`);
      pendingActionRef.current = action;
    };
  }, [speak]);

  // 전역 더블클릭 핸들러
  const handleGlobalDoubleClick = useCallback(() => {
    if (pendingActionRef.current) {
      pendingActionRef.current();
      pendingActionRef.current = null;
    }
  }, []);

  return (
    <div className="home-container" onDoubleClick={handleGlobalDoubleClick}>
      <header className="home-header">
        <h1>👁️ 시각 도우미</h1>
        <p className="subtitle">원하는 기능을 선택하세요</p>

        {isSpeaking && (
          <button onClick={stopSpeaking} className="stop-speaking-btn">
            ⏹️ 음성 중지
          </button>
        )}
      </header>

      <main className="home-main">
        <div className="service-buttons">
          <button
            onClick={handleSelection('실시간 물체 인식', () => navigate('/object-detection'))}
            className="service-btn service-btn-primary"
          >
            <div className="service-icon">📹</div>
            <div className="service-title">실시간 물체 인식</div>
            <div className="service-desc">카메라로 주변 물체와 거리를 실시간으로 안내합니다</div>
          </button>

          <button
            onClick={handleSelection('사진 영상 질문하기', () => navigate('/vision-assistant'))}
            className="service-btn service-btn-secondary"
          >
            <div className="service-icon">📷</div>
            <div className="service-title">사진·영상 질문하기</div>
            <div className="service-desc">사진이나 영상을 업로드하고 궁금한 점을 물어보세요</div>
          </button>
        </div>

        <div className="help-section">
          <button
            onClick={handleSelection('사용법 듣기', () => {
              speak(`
                시각 도우미 앱입니다.
                첫 번째는 실시간 물체 인식 기능으로, 카메라를 통해 주변 물체와 거리를 실시간으로 안내합니다.
                두 번째는 사진 영상 질문하기 기능으로, 사진이나 영상을 업로드하고 궁금한 점을 물어볼 수 있습니다.
                원하는 기능을 선택한 후 화면을 두 번 두드리면 해당 기능으로 이동합니다.
              `);
            })}
            className="help-btn"
          >
            ❓ 사용법 듣기
          </button>
        </div>
      </main>

      <footer className="home-footer">
        <p>버튼 선택 후 화면 더블탭하여 실행</p>
      </footer>
    </div>
  );
}

export default Home;
