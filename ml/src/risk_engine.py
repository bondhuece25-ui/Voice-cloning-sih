from collections import deque


LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70
ROLLING_WINDOW_SIZE = 5


class RiskEngine:
    """Maintain a rolling spoof risk score and produce a decision."""

    def __init__(
        self,
        low_threshold: float = LOW_RISK_THRESHOLD,
        high_threshold: float = HIGH_RISK_THRESHOLD,
        window_size: int = ROLLING_WINDOW_SIZE,
    ) -> None:
        if not 0.0 <= low_threshold <= 1.0:
            raise ValueError("Low threshold must be between 0 and 1.")

        if not 0.0 <= high_threshold <= 1.0:
            raise ValueError("High threshold must be between 0 and 1.")

        if low_threshold >= high_threshold:
            raise ValueError("Low threshold must be lower than high threshold.")

        if window_size <= 0:
            raise ValueError("Window size must be positive.")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.scores: deque[float] = deque(maxlen=window_size)

    def update(self, spoof_probability: float) -> dict[str, float | str]:
        """Add one spoof probability and return the current risk decision."""

        if not 0.0 <= spoof_probability <= 1.0:
            raise ValueError("Spoof probability must be between 0 and 1.")

        self.scores.append(float(spoof_probability))

        rolling_risk = sum(self.scores) / len(self.scores)

        if rolling_risk >= self.high_threshold:
            decision = "BLOCK"
        elif rolling_risk >= self.low_threshold:
            decision = "WARNING"
        else:
            decision = "ALLOW"

        return {
            "spoof_probability": float(spoof_probability),
            "rolling_risk": float(rolling_risk),
            "decision": decision,
        }