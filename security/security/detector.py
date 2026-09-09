import os
import sys
import joblib
import numpy as np

# Allow importing feature_extractor.py from the security folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import extract_features


# -----------------------------
# MODEL PATH
# -----------------------------

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "voice_spoof_model.pkl"
)


# -----------------------------
# VOICE DETECTION FUNCTION
# -----------------------------

def detect_voice(audio_file):

    # Check model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Trained model not found: " + MODEL_FILE
        )

    # Check audio file
    if not os.path.exists(audio_file):
        raise FileNotFoundError(
            "Audio file not found: " + audio_file
        )

    # Load trained model
    model = joblib.load(MODEL_FILE)
    print("Model classes:", model.classes_)

    if hasattr(model, "n_features_in_"):
      print("Model feature count:", model.n_features_in_)

    # Extract features
    features = extract_features(audio_file)

    # Convert to numpy array
    features = np.asarray(features)

    # Reshape for the model
    features = features.reshape(1, -1)

    # -----------------------------
    # DEBUG INFORMATION
    # -----------------------------

    print()
    print("Features shape:", features.shape)

    if hasattr(model, "n_features_in_"):
        print("Model expects:", model.n_features_in_)

    # -----------------------------
    # FEATURE CHECK
    # -----------------------------

    if hasattr(model, "n_features_in_"):

        expected = model.n_features_in_
        actual = features.shape[1]

        if actual != expected:
            raise ValueError(
                f"Feature mismatch: model expects {expected} "
                f"features, but detector produced {actual} features."
            )

    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction = model.predict(features)[0]

    # -----------------------------
    # CONFIDENCE
    # -----------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)[0]

        print("Model classes:", model.classes_)
        print("Probabilities:", probabilities)

        confidence = float(np.max(probabilities))

    else:

        confidence = 1.0

    # -----------------------------
    # RESULT
    # -----------------------------

    if prediction == 0:
        result = "GENUINE"
    else:
        result = "SPOOFED"

    return result, confidence


# -----------------------------
# RUN DIRECTLY
# -----------------------------

if __name__ == "__main__":

    print("=" * 45)
    print("       VOICE SPOOF DETECTOR")
    print("=" * 45)

    audio_file = input("Enter audio file path: ").strip()

    # Remove quotes if the user pasted a quoted path
    audio_file = audio_file.strip('"').strip("'")

    if not os.path.exists(audio_file):

        print()
        print("Audio file not found!")
        print("Check the path and try again.")
        sys.exit(1)

    try:

        result, confidence = detect_voice(audio_file)

        print()
        print("=" * 45)
        print("Detection Result:", result)
        print("Confidence:", f"{confidence * 100:.2f}%")
        print("=" * 45)

    except Exception as e:

        print()
        print("=" * 45)
        print("ERROR")
        print("=" * 45)
        print(e)
        print("=" * 45)