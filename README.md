## Machine Learning – Poojitha (ML Lead)

### Role
Responsible for the machine learning pipeline for real-time voice-cloning
impersonation detection.

### Responsibilities
- Dataset preparation and preprocessing
- Audio standardization to 16 kHz mono
- Development and training of the voice anti-spoofing model
- Model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC
- Local model inference and prediction testing
- Development of rolling spoof-risk analysis
- Real-time detection design using short audio windows
- Backend-facing ML API for spoof probability and risk decisions
- ML integration support and testing with the backend team

### ML Pipeline

Audio Input
→ Audio Preprocessing
→ Wav2Vec2 Feature Extraction
→ Binary Classification
→ Spoof Probability
→ Rolling Risk Analysis
→ ALLOW / WARNING / BLOCK

### Model
- Backbone: Wav2Vec2
- Classification: Binary (BONAFIDE / SPOOF)
- Sample Rate: 16 kHz
- Input: Mono audio
- Framework: PyTorch + Hugging Face Transformers

### Current Evaluation
- Test Accuracy: 95.99%
- ROC-AUC: 98.57%

### Real-Time Detection
The system is designed to process audio in short windows rather than waiting
for an entire call to finish. Model predictions are converted into a rolling
spoof-risk score, which can trigger:

- ALLOW – low spoof risk
- WARNING – suspicious audio
- BLOCK – high spoof risk

### ML–Backend Interface
The ML module provides the backend with:
- Spoof probability
- Rolling risk score
- Decision

This allows the backend to use the ML prediction for real-time call-level
decision making.
