import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import ObjectDetection from './pages/ObjectDetection';
import VisionAssistant from './pages/VisionAssistant';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/object-detection" element={<ObjectDetection />} />
        <Route path="/vision-assistant" element={<VisionAssistant />} />
      </Routes>
    </Router>
  );
}

export default App;
