class PlanningAgent:

    name = "Planning / Decision Agent"

    def decide(
        self,
        predictive_result,
        knowledge_result,
        vision_result
    ):

        failure_probability = predictive_result.get(
            "failure_probability", 0
        )

        risk_level = predictive_result.get(
            "risk_level", "LOW"
        )

        vision_data = vision_result.get(
            "result", {}
        )

        severity = vision_data.get(
            "severity", "Unknown"
        )

        if risk_level == "HIGH":

            recommendation = (
                "Inspect machine before returning "
                "it to full production."
            )

            priority = "HIGH"

        elif risk_level == "MEDIUM":

            recommendation = (
                "Schedule preventive inspection "
                "and continue monitoring."
            )

            priority = "MEDIUM"

        else:

            recommendation = (
                "Continue operation with routine monitoring."
            )

            priority = "LOW"

        return {

            "agent": self.name,

            "risk_level": risk_level,

            "failure_probability": failure_probability,

            "vision_severity": severity,

            "priority": priority,

            "recommendation": recommendation,

            "evidence": knowledge_result.get(
                "evidence", []
            )
        }
