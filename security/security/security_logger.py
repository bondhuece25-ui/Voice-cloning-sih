import os
from datetime import datetime


LOG_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs"
)

LOG_FILE = os.path.join(
    LOG_FOLDER,
    "security_events.log"
)


os.makedirs(LOG_FOLDER, exist_ok=True)


def log_security_event(result, confidence, risk_level, action, message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log_entry = (
        f"[{timestamp}] "
        f"RESULT={result} | "
        f"CONFIDENCE={confidence * 100:.2f}% | "
        f"RISK={risk_level} | "
        f"ACTION={action} | "
        f"MESSAGE={message}\n"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(log_entry)