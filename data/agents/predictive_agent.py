class PredictiveMaintenanceAgent:

    name = "Predictive Maintenance Agent"

    def analyze(self, failure_probability):

        if failure_probability >= 0.70:
            risk = "HIGH"
        elif failure_probability >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "agent": self.name,
            "failure_probability": round(failure_probability, 3),
            "risk_level": risk
        }
