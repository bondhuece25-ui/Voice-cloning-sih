import "./Home.css";

function Home({
  onStartProtection,
  onAnalyzeAudio,
})  {
  return (
    <div className="home-page">

      <header className="home-header">
        <div>
          <h2 className="brand-name">VoiceShield AI</h2>
          <p className="brand-subtitle">
            Real-Time Voice Clone Detection & Prevention
          </p>
        </div>

        <div className="sih-badge">
          SIH 2026
        </div>
      </header>

      <main className="hero">

        <p className="hero-label">
          AI-POWERED VOICE SECURITY
        </p>

        <h1 className="hero-title">
          Protect yourself from
          <br />
          <span className="hero-highlight">
            AI voice impersonation.
          </span>
        </h1>

        <p className="hero-description">
          VoiceShield analyzes voice conversations in real time to identify
          potential AI-generated voices and help users respond before fraud
          happens.
        </p>

        <div className="hero-actions">
          <button
              className="primary-button"
              onClick={onStartProtection}
         >
  Start Protection
</button>

          <button
                className="secondary-button"
                onClick={onAnalyzeAudio}
          >
              Analyze Audio
          </button>
        </div>
        <div className="protection-card">
  <div className="protection-header">
    <div>
      <p className="protection-label">PROTECTION STATUS</p>
      <h3>VoiceShield is active</h3>
    </div>

    <div className="live-status">
      <span className="live-dot"></span>
      LIVE
    </div>
  </div>

  <div className="protection-body">
    <div className="waveform">
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
    </div>

    <p>
      Monitoring voice communication for potential AI-generated speech.
    </p>
  </div>
</div>
        <div className="trust-row">

          <div className="trust-item">
            <span className="trust-dot"></span>
            Real-Time Analysis
          </div>

          <div className="trust-item">
            <span className="trust-dot"></span>
            AI-Powered Detection
          </div>

          <div className="trust-item">
            <span className="trust-dot"></span>
            Privacy Focused
          </div>

        </div>

      </main>

    </div>
  );
}

export default Home;