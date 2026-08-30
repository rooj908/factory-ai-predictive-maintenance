class VisionAgent:

    name = "Vision Agent"

    def analyze(self, image_path=None):

        if image_path:

            result = {
                "defect": "Mechanical anomaly",
                "severity": "Medium",
                "confidence": 0.85,
                "image": image_path
            }

        else:

            result = {
                "defect": "No image provided",
                "severity": "Unknown",
                "confidence": 0.0,
                "image": None
            }

        return {
            "agent": self.name,
            "result": result
        }
