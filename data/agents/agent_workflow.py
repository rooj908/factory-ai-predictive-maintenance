from agents.predictive_agent import PredictiveMaintenanceAgent
from agents.vision_agent import VisionAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.planning_agent import PlanningAgent


class FactoryAgentWorkflow:

    def __init__(self):

        self.predictive_agent = PredictiveMaintenanceAgent()
        self.vision_agent = VisionAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.planning_agent = PlanningAgent()

    def run(
        self,
        failure_probability,
        image_path=None,
        query="machine has high vibration and needs maintenance"
    ):

        # Agent 1: Predictive Maintenance
        predictive_result = self.predictive_agent.analyze(
            failure_probability
        )

        # Agent 2: Vision
        vision_result = self.vision_agent.analyze(
            image_path
        )

        # Agent 3: Knowledge / RAG
        knowledge_result = self.knowledge_agent.analyze(
            query
        )

        # Agent 4: Planning / Decision
        decision = self.planning_agent.decide(
            predictive_result,
            knowledge_result,
            vision_result
        )

        return {
            "predictive_agent": predictive_result,
            "vision_agent": vision_result,
            "knowledge_agent": knowledge_result,
            "planning_agent": decision
        }


if __name__ == "__main__":

    workflow = FactoryAgentWorkflow()

    result = workflow.run(
        failure_probability=0.78,
        image_path=None
    )

    print("=" * 60)
    print("MULTI-AGENT FACTORY WORKFLOW")
    print("=" * 60)

    for agent_name, output in result.items():

        print("\n" + agent_name.upper())
        print("-" * 40)
        print(output)
