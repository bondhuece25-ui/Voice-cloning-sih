from typing import Any

from .risk_engine import RiskEngine


class AntiSpoofAPI:
    """
    Backend-facing interface for the voice anti-spoofing system.

    Input:
        spoof_probability: probability produced by the ML model

    Output:
        spoof_probability
        rolling_risk
        decision
    """

    def __init__(self) -> None:
        self.risk_engine = RiskEngine()

    def process_model_probability(
        self,
        spoof_probability: float,
    ) -> dict[str, Any]:

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