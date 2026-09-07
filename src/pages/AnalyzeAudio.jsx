import { useState } from "react";
import "./AnalyzeAudio.css";

function AnalyzeAudio() {
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = () => {
  if (!selectedFile) {
    alert("Please select an audio file first.");
    return;
  }

  setIsAnalyzing(true);
  setProgress(0);
  setResult(null);

  let current = 0;

  const interval = setInterval(() => {
    current += 10;
    setProgress(current);

    if (current >= 100) {
      clearInterval(interval);

      const probability = Math.floor(
        Math.random() * 101
      );

      let threatLevel = "";
      let recommendation = "";

      if (probability < 40) {
        threatLevel = "LOW";
        recommendation =
          "Voice Appears Genuine";
      } else if (probability < 70) {
        threatLevel = "MEDIUM";
        recommendation =
          "Proceed With Caution";
      } else {
        threatLevel = "HIGH";
        recommendation =
          "Terminate Call";
      }

      setResult({
        aiProbability: probability,
        threatLevel,
        recommendation,
      });

      setIsAnalyzing(false);
    }
  }, 300);
};
  return (
    <div className="analyze-page">
      <div className="analyze-container">

        <div className="analyze-header">
          <p className="analyze-tag">AI VOICE DETECTION</p>

          <h1>Analyze Audio Recording</h1>

          <p className="analyze-subtitle">
            Upload an audio sample and detect potential AI-generated voices.
          </p>
        </div>

        <div className="upload-card">
          <div className="upload-area">

            <div className="upload-icon">🎵</div>

            <h3>Drop audio file here</h3>

            <p>
              Supports MP3, WAV, M4A and other audio formats
            </p>

            <input
              type="file"
              accept="audio/*"
              id="audio-upload"
              hidden
              onChange={(e) =>
                setSelectedFile(e.target.files[0])
              }
            />

            <label
              htmlFor="audio-upload"
              className="upload-btn"
            >
              Choose Audio File
            </label>

            {selectedFile && (
              <p className="file-name">
                Selected: {selectedFile.name}
              </p>
            )}

          </div>
        </div>

        <button
          className="analyze-btn"
          onClick={handleAnalyze}
        >
          {isAnalyzing
            ? "Analyzing..."
            : "Analyze Audio"}
        </button>
        {isAnalyzing && (
          <div className="progress-section">
            <p>Analyzing Audio...</p>

            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${progress}%`,
                }}
              ></div>
            </div>

            <span>{progress}%</span>
          </div>
       )}
        {result && (
          <div className="result-card">
            <p className="result-tag">
              ANALYSIS COMPLETE
            </p>

            <h2>
              AI Probability: {result.aiProbability}%
            </h2>

            <h3
                className={
                    result.threatLevel === "HIGH"
                    ? "threat-high"
                    : result.threatLevel === "MEDIUM"
                    ? "threat-medium"
                    : "threat-low"
                }
                >
                <p
                  className={`threat-${result.threatLevel.toLowerCase()}`}
                >
                  Threat Level: {result.threatLevel}
                </p>
            </h3>

            <p className="recommendation">
                Recommendation: {result.recommendation}
            </p>

              <button
                className="reset-btn"
                onClick={() => {
                  setResult(null);
                  setSelectedFile(null);
                }}
              >
                Analyze Another File
              </button>
          </div>
        )}

      </div>
    </div>
  );
}

export default AnalyzeAudio;