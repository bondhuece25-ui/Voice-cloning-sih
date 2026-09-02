import "./Protection.css";

function Protection({ onEndProtection }) {
  return (
    <div className="protection-page">
      <header className="protection-header">
        <div>
          <p className="protection-eyebrow">VOICE PROTECTION</p>
          <h1>Call Protection</h1>
        </div>

        <div className="protection-live">
          <span className="protection-live-dot"></span>
          PROTECTION ACTIVE
        </div>
      </header>

      <main className="protection-content">
        <section className="call-card">
          <div className="call-top">
            <div>
              <p className="section-label">CURRENT CALL</p>
              <h2>Unknown Caller</h2>
              <p className="caller-number">+91 ••••• •••••</p>
            </div>

            <div className="call-timer">02:14</div>
          </div>

          <div className="voice-status">
            <div className="voice-status-icon">🎙️</div>

            <div>
              <p className="section-label">VOICE ANALYSIS</p>
              <h3>Analyzing incoming voice</h3>
              <p>
                VoiceShield is continuously checking the conversation for
                signs of synthetic or cloned speech.
              </p>
            </div>
          </div>

          <div className="risk-section">
            <div className="risk-heading">
              <div>
                <p className="section-label">CURRENT RISK</p>
                <h3>Low Risk</h3>
              </div>

              <strong>24%</strong>
            </div>

            <div className="risk-bar">
              <div className="risk-fill"></div>
            </div>

            <p className="risk-note">
              No significant indicators of voice impersonation detected.
            </p>
          </div>
        </section>

        <section className="analysis-grid">
          <div className="info-card">
            <p className="section-label">VOICE AUTHENTICITY</p>
            <h3>76%</h3>
            <p>Confidence that the voice is human-generated.</p>
          </div>

          <div className="info-card">
            <p className="section-label">ANALYSIS STATUS</p>
            <h3>Live</h3>
            <p>Audio stream is being analyzed continuously.</p>
          </div>

          <div className="info-card">
            <p className="section-label">THREAT LEVEL</p>
            <h3 className="safe-text">Low</h3>
            <p>No immediate action is required.</p>
          </div>
        </section>

        <button
           className="end-protection"
           onClick={onEndProtection}
           >
           End Protection
        </button>
      </main>
    </div>
  );
}

export default Protection;