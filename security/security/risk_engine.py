def security_decision(result, confidence):

    confidence_percent = confidence * 100

    if (
        result == "SPOOFED"
        and confidence_percent >= 80
    ):

        return {
            "risk_level": "HIGH",
            "action": "BLOCK",
            "message": "Potential voice impersonation detected."
        }

    elif (
        result == "GENUINE"
        and confidence_percent >= 80
    ):

        return {
            "risk_level": "LOW",
            "action": "ALLOW",
            "message": "Voice appears genuine."
        }

    else:

        return {
            "risk_level": "MEDIUM",
            "action": "VERIFY",
            "message": "Voice authenticity could not be confirmed."
        }