from typing import Any

from risk_engine import RiskEngine


class AntiSpoofAPI:
    """
    Backend-facing interface for the voice anti-spoofing system.

    The backend will eventually provide:
        - audio chunks
        - sample rate

    This class returns:
        - current spoof probability
        - rolling risk
        - decision
    """

    def __init__(self) -> None:
        self.risk_engine = RiskEngine()

    def process_model_probability(
        self,
        spoof_probability: float,
    ) -> dict[str, Any]:
        """
        Receive one model prediction and convert it
        into a backend-friendly result.
        """

        result = self.risk_engine.update(
            spoof_probability
        )

        return {
            "spoof_probability": round(
                float(result["spoof_probability"]),
                4,
            ),
            "rolling_risk": round(
                float(result["rolling_risk"]),
                4,
            ),
            "decision": result["decision"],
        }