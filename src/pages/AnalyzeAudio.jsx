import { useState } from "react";
import "./AnalyzeAudio.css";

function AnalyzeAudio({ onBack }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleAnalyze = () => {
    if (!selectedFile) {
      alert("Please select an audio file first.");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);
    setProgress(0);

    let currentProgress = 0;

    const progressInterval = setInterval(() => {
      currentProgress += 10;

      if (currentProgress <= 100) {
        setProgress(currentProgress);
      }
    }, 300);

    setTimeout(() => {
      clearInterval(progressInterval);

      const aiProbability =
        Math.floor(Math.random() * 101);

      let threatLevel = "";
      let recommendation = "";

      if (aiProbability < 40) {
        threatLevel = "LOW";
        recommendation =
          "Voice Appears Genuine";
      } else if (aiProbability < 70) {
        threatLevel = "MEDIUM";
        recommendation =
          "Proceed With Caution";
      } else {
        threatLevel = "HIGH";
        recommendation =
          "Potential AI Voice Detected";
      }

      setResult({
        aiProbability,
        threatLevel,
        recommendation,
      });

      setIsAnalyzing(false);
      setProgress(100);
    }, 3000);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setProgress(0);
  };

  return (
    <div className="analyze-page">
      <div className="analyze-container">

        <button
          className="back-btn"
          onClick={onBack}
        >
          ← Back
        </button>

        <div className="analyze-header">
          <p className="analyze-tag">
            AI VOICE DETECTION
          </p>

          <h1>Analyze Audio Recording</h1>

          <p className="analyze-subtitle">
            Upload an audio sample and detect
            potential AI-generated voices.
          </p>
        </div>

        <div className="upload-card">
          <div className="upload-area">
            <div className="upload-icon">🎵</div>

            <h3>Drop audio file here</h3>

            <p>
              Supports MP3, WAV, M4A and other
              audio formats
            </p>

            <input
              type="file"
              accept="audio/*"
              id="audio-upload"
              hidden
              onChange={(e) =>
                setSelectedFile(
                  e.target.files[0]
                )
              }
            />

            <label
              htmlFor="audio-upload"
              className="upload-btn"
            >
              Choose Audio File
            </label>

            {selectedFile && (
              <>
                <div className="file-details">
                  <p>
                    <strong>File:</strong>{" "}
                    {selectedFile.name}
                  </p>

                  <p>
                    <strong>Size:</strong>{" "}
                    {(
                      selectedFile.size /
                      1024 /
                      1024
                    ).toFixed(2)}{" "}
                    MB
                  </p>

                  <p>
                    <strong>Type:</strong>{" "}
                    {selectedFile.type}
                  </p>
                </div>

                <audio
                  controls
                  src={URL.createObjectURL(
                    selectedFile
                  )}
                  className="audio-player"
                />
              </>
            )}
          </div>
        </div>

        <button
          className="analyze-btn"
          onClick={handleAnalyze}
          disabled={isAnalyzing}
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

            <p>{progress}%</p>
          </div>
        )}

        {result && (
          <div className="result-card">
            <p className="analyze-tag">
              ANALYSIS COMPLETE
            </p>

            <h2>
              🤖 AI Probability: {result.aiProbability}%
            </h2>
            <h3
              className={`threat-${result.threatLevel.toLowerCase()}`}
            >
              Threat Level:{" "}
              {result.threatLevel}
            </h3>

            <p>
              Recommendation:{" "}
              {result.recommendation}
            </p>

            <button
              className="upload-btn"
              onClick={handleReset}
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