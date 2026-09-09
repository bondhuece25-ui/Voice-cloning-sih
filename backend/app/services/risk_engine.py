from collections import deque


def calculate_risk(score: float) -> str:
    """
    Convert a spoof probability into a risk level.
    """

    if not 0.0 <= score <= 1.0:
        raise ValueError("Prediction score must be between 0 and 1")

    if score < 0.3:
        return "LOW"

    elif score < 0.7:
        return "MEDIUM"

    elif score < 0.9:
        return "HIGH"

    return "CRITICAL"


class TemporalRiskEngine:
    """
    Maintains a sliding window of recent prediction scores
    and calculates a more stable risk level.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)

    def add_score(self, score: float) -> dict:
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Prediction score must be between 0 and 1"
            )

        self.scores.append(score)

        average_score = sum(self.scores) / len(self.scores)

        risk = calculate_risk(average_score)

        return {
            "current_score": score,
            "average_score": average_score,
            "risk_level": risk,
            "samples": len(self.scores)
        }

    def reset(self):
        self.scores.clear()