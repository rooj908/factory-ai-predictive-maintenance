# Factory AI Predictive Maintenance System

AI-powered industrial predictive maintenance system combining Machine Learning, Explainable AI, Computer Vision, RAG, Multi-Agent AI, Digital Twin simulation, and Human-in-the-Loop decision making.

## Project Overview

Factory AI analyzes machine sensor data to predict equipment failure risk and recommend maintenance actions.

### System Workflow

Machine Sensor Data
        ?
Predictive ML Model
        ?
Failure Risk Prediction
        ?
Explainable AI
        ?
RAG Knowledge Retrieval
        ?
Multi-Agent AI Workflow
        ?
Computer Vision
        ?
Digital Twin Simulation
        ?
Human Supervisor Approval
        ?
Final Maintenance Decision

## Features

- Predictive machine failure detection
- Random Forest classification
- Feature importance and XAI
- RAG maintenance knowledge retrieval
- Groq LLM maintenance explanations
- Multi-agent AI workflow
- Computer Vision module
- Machine image upload
- Digital Twin what-if analysis
- Human-in-the-loop decision making
- MLflow experiment tracking
- Streamlit dashboard

## Machine Learning Performance

| Metric | Result |
|---|---:|
| Accuracy | ~98% |
| ROC-AUC | ~0.997 |
| Failure Recall | ~0.96 |
| Test Samples | 4,809 |

### Important Features

1. Motor Temperature
2. RPM
3. Average Phase Current
4. Vibration RMS
5. Hours Since Maintenance
6. Pressure Level
7. Ambient Temperature

## AI Agents

### Predictive Maintenance Agent
Predicts failure probability and assigns LOW, MEDIUM, or HIGH risk.

### Vision Agent
Processes uploaded machine images and provides a visual inspection result.

### Knowledge Agent
Retrieves relevant maintenance information using RAG.

### Planning / Decision Agent
Combines prediction, vision and knowledge evidence to recommend an action.

## RAG Knowledge Base

Example sources:

- High Vibration Maintenance Guide
- Preventive Maintenance SOP
- Electrical Safety SOP

Example recommendation:

"Inspect machine before returning it to full production."

## Digital Twin

The Digital Twin compares operational scenarios based on failure risk, downtime, production loss and estimated cost.

Example scenarios:

| Scenario | Downtime | Failure Risk | Estimated Cost |
|---|---:|---:|---:|
| Continue Operation | 0h | 78% | 3900 |
| Stop for Maintenance | 4h | 15% | 8750 |
| Reduce Machine Load | 1h | 35% | 3750 |

## Human-in-the-Loop

The system allows a human supervisor to:

- APPROVE
- REJECT
- MODIFY

AI recommendations are treated as decision support. Final operational authority remains with the human supervisor.

## MLflow

Experiment:

Factory Predictive Maintenance

Tracked metrics include:

- Accuracy
- ROC-AUC
- Failure Recall
- Model artifacts

## Project Structure

factory_ai/
¦
+-- agents/
¦   +-- __init__.py
¦   +-- predictive_agent.py
¦   +-- vision_agent.py
¦   +-- knowledge_agent.py
¦   +-- planning_agent.py
¦   +-- agent_workflow.py
¦   +-- human_decision.py
¦
+-- app/
¦   +-- streamlit_app.py
¦
+-- data/
¦   +-- factory_data.csv
¦   +-- images/
¦       +-- machine_bw.png
¦
+-- models/
+-- rag/
+-- xai/
+-- reports/
¦
+-- main.py
+-- digital_twin.py
+-- mlflow_train.py
+-- config.py
+-- requirements.txt
+-- .gitignore
+-- README.md

## Installation

Clone the repository:

git clone https://github.com/rooj908/factory-ai-predictive-maintenance.git

Enter the project:

cd factory-ai-predictive-maintenance

Install dependencies:

pip install -r requirements.txt

## Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key

Never commit API keys or the .env file to GitHub.

## Run Streamlit Dashboard

streamlit run app/streamlit_app.py

Open:

http://localhost:8501

## Run Components

Predictive Maintenance:

python main.py

RAG:

python rag/retriever.py

Multi-Agent Workflow:

python -m agents.agent_workflow

Human Supervisor:

python agents/human_decision.py

MLflow:

python mlflow_train.py

Digital Twin:

python digital_twin.py

## Safety

This project is designed as an AI decision-support system. Industrial maintenance actions should be reviewed by qualified personnel before implementation.

## Future Improvements

- Real-time IoT sensor integration
- Real computer vision defect detection
- Live sensor streaming
- Advanced anomaly detection
- Cloud deployment
- Automated maintenance scheduling
- SCADA/PLC integration

## Author

Urooj Fatima and Unsha

GitHub:
https://github.com/rooj908

## Hackathon Project

Factory AI: Intelligent Predictive Maintenance & Decision Support System
