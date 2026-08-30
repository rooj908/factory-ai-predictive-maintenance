# 🏭 Factory AI: Intelligent Predictive Maintenance & Decision Support System

**AI-powered industrial predictive maintenance and decision-support system**

Factory AI is an intelligent industrial maintenance system that combines **Machine Learning, Deep Learning, Explainable AI, Computer Vision, RAG, Multi-Agent AI, MLflow, and Digital Twin simulation** into one interactive Streamlit dashboard.

The system analyzes machine sensor data to identify potential failure risk, explains important prediction factors, retrieves relevant maintenance knowledge, supports visual machine inspection, compares maintenance scenarios, and provides recommendations for human review.

---

## 🏆 Hackathon Project

**Project:** Factory AI: Intelligent Predictive Maintenance & Decision Support System

**Developed by:** Urooj Fatima & Unsha Siddiqui

This project was developed as a hackathon solution demonstrating how modern AI techniques can support industrial predictive maintenance and maintenance decision-making.

---

## 🎯 Problem Statement

Unexpected industrial machine failures can cause:

* Production downtime
* Equipment damage
* High maintenance costs
* Reduced productivity
* Safety risks

Traditional maintenance approaches are often reactive or based on fixed schedules.

### 💡 Our Solution

Factory AI uses machine sensor data and AI-based decision support to identify potential machine failure risk **before a breakdown occurs**.

The system helps answer:

> **"Is this machine at risk, why is it at risk, and what should we do next?"**

---

# 🧠 System Architecture

```text
                    ┌─────────────────────────┐
                    │     Machine Sensors     │
                    │                         │
                    │ Temperature / RPM       │
                    │ Vibration / Pressure    │
                    │ Current / Maintenance    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Data Analysis & EDA   │
                    │                         │
                    │ Cleaning / Statistics    │
                    │ Correlation / Outliers   │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │        Predictive Models           │
              │                                    │
              │ Random Forest + Deep Learning MLP  │
              └────────────────┬───────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │      Failure Risk       │
                    │                         │
                    │ LOW / MEDIUM / HIGH     │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌────────────┐    ┌────────────┐    ┌────────────┐
       │    XAI     │    │    RAG     │    │   Vision   │
       │            │    │            │    │            │
       │ Explain    │    │ Maintenance│    │  Machine   │
       │ Prediction │    │ Knowledge  │    │ Inspection │
       └──────┬─────┘    └──────┬─────┘    └──────┬─────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │    Multi-Agent System   │
                    │                         │
                    │ Predictive Agent        │
                    │ Vision Agent            │
                    │ Knowledge Agent         │
                    │ Planning Agent          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Digital Twin       │
                    │                         │
                    │ What-if Simulation      │
                    │ Risk / Cost / Downtime  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Human Supervisor      │
                    │                         │
                    │ APPROVE / REJECT /      │
                    │ MODIFY                  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Final Maintenance       │
                    │ Decision                │
                    └─────────────────────────┘
```

---

# ⚙️ How It Works

## 1. Machine Data

The system uses industrial machine and operational features including:

* Motor temperature
* RPM
* Vibration RMS
* Average phase current
* Pressure level
* Hours since maintenance
* Ambient temperature
* Machine type
* Operating mode

---

## 2. Data Analysis

The project includes exploratory data analysis covering:

* Dataset inspection
* Data types
* Missing-value analysis
* Statistical analysis
* Distribution analysis
* Outlier investigation
* Correlation analysis
* Feature relationships
* Failure-related analysis

---

## 3. Predictive Machine Learning

A **Random Forest Classifier** is used to predict whether a machine is likely to experience failure within the next 24 hours.

The model uses preprocessing for:

* Numerical features
* Categorical features
* One-hot encoding
* Unknown categorical values

The model also uses class balancing to improve handling of failure cases.

---

## 4. Deep Learning

Factory AI also includes a trained **MLP Neural Network** using TensorFlow/Keras.

The Deep Learning model predicts:

> **Probability of machine failure within the next 24 hours**

The Streamlit application automatically uses the Deep Learning prediction when the trained model and preprocessor are available.

### Deep Learning Model

```text
Input Features
      ↓
Preprocessing
      ↓
MLP Neural Network
      ↓
Failure Probability
      ↓
Risk Classification
```

Reported model performance:

* **Test Accuracy:** 95.13%
* **ROC-AUC:** 0.9827

---

## 5. Risk Assessment

The predicted probability is converted into three operational risk levels:

```text
Probability < 40%
        ↓
      LOW

40% - 69%
        ↓
     MEDIUM

Probability >= 70%
        ↓
      HIGH
```

---

# 🔍 Explainable AI

Factory AI is designed to provide more than a simple prediction.

The XAI component helps identify important factors contributing to machine failure predictions.

Important machine features include:

1. Motor Temperature
2. RPM
3. Average Phase Current
4. Vibration RMS
5. Hours Since Maintenance
6. Pressure Level
7. Ambient Temperature

Instead of only showing:

```text
Failure = YES
```

the system aims to provide additional information about **why the machine is considered risky**.

---

# 📚 RAG Maintenance Knowledge

Factory AI includes a Retrieval-Augmented Generation workflow for maintenance knowledge.

The Knowledge Agent can retrieve relevant information from the project's maintenance knowledge base.

Example knowledge areas include:

* High Vibration Maintenance Guide
* Preventive Maintenance SOP
* Electrical Safety SOP

### Example Workflow

```text
Machine Condition
       ↓
High Vibration
       +
Elevated Temperature
       ↓
Retrieve Relevant Knowledge
       ↓
Maintenance Guidance
       ↓
AI Recommendation
```

---

# 👁️ Computer Vision

The Streamlit dashboard allows users to upload machine images.

The Vision Agent can be used as part of the maintenance assessment to provide visual inspection information.

The dashboard displays:

* Uploaded machine image
* Detected defect/anomaly
* Severity
* Confidence

This allows sensor-based predictive maintenance to be combined with visual inspection.

---

# 🤖 Multi-Agent AI System

Factory AI uses multiple specialized agents.

## 🔮 Predictive Maintenance Agent

Analyzes predictive failure information and provides:

* Failure probability
* Risk level
* Predictive maintenance information

---

## 👁️ Vision Agent

Processes uploaded machine images and provides visual inspection results.

---

## 📚 Knowledge Agent

Retrieves relevant maintenance information using the RAG workflow.

---

## 🧠 Planning / Decision Agent

Combines information from:

* Predictive model
* Deep Learning
* Vision
* Maintenance knowledge
* Risk assessment

and generates a maintenance recommendation.

---

# 🪞 Digital Twin Simulation

The Digital Twin component provides **what-if analysis**.

Users can compare different operational strategies before making a maintenance decision.

The system evaluates:

* Failure risk
* Downtime
* Production loss
* Estimated cost

### Example Scenarios

| Scenario             | Downtime | Failure Risk | Production Loss | Estimated Cost |
| -------------------- | -------: | -----------: | --------------: | -------------: |
| Continue Operation   |       0h |          78% |               0 |         $3,900 |
| Stop for Maintenance |       4h |          15% |             400 |         $8,750 |
| Reduce Machine Load  |       1h |          35% |             100 |         $3,750 |

The purpose of the Digital Twin is to help maintenance teams compare possible actions rather than relying only on a single prediction.

---

# 👨‍💼 Human-in-the-Loop

Factory AI is a **decision-support system**, not an autonomous industrial control system.

The AI provides recommendations, but the final decision remains with a qualified human supervisor.

```text
┌────────────┐
│   APPROVE  │
└────────────┘

┌────────────┐
│   REJECT   │
└────────────┘

┌────────────┐
│   MODIFY   │
└────────────┘
```

Actual maintenance operations should always follow appropriate industrial safety procedures.

---

# 📊 Machine Learning Performance

The current predictive model achieved approximately:

| Metric         | Result |
| -------------- | -----: |
| Accuracy       |   ~98% |
| ROC-AUC        | ~0.997 |
| Failure Recall |  ~0.96 |
| Test Samples   |  4,809 |

> Model performance may vary depending on data splits, preprocessing, and retraining configuration.

---

# 🖥️ Streamlit Dashboard

Factory AI includes an interactive Streamlit dashboard.

### Dashboard Tabs

| Tab                | Function                        |
| ------------------ | ------------------------------- |
| 📊 Prediction      | Machine failure prediction      |
| 🧠 Deep Learning   | Neural network prediction       |
| 📷 Computer Vision | Machine image inspection        |
| 📚 RAG Knowledge   | Maintenance knowledge retrieval |
| 🤖 Agents          | Multi-agent decision workflow   |
| 🔮 Digital Twin    | What-if maintenance simulation  |

### Dashboard Features

* Machine sensor input
* Failure probability
* Risk classification
* Deep Learning prediction
* Machine image upload
* Computer Vision inspection
* RAG knowledge retrieval
* AI maintenance recommendation
* Digital Twin simulation
* Human decision support

---

# 📸 Dashboard Preview

Screenshots can be added under:

```text
docs/
└── screenshots/
    ├── dashboard.png
    ├── prediction.png
    ├── deep_learning.png
    ├── computer_vision.png
    ├── rag.png
    ├── agents.png
    └── digital_twin.png
```

Example:

```markdown
![Factory AI Dashboard](docs/screenshots/dashboard.png)
```

---

# 🎥 Demo

## Live Demo

Add the deployed Streamlit application URL here after deployment:

```text
Coming Soon
```

## Demo Video

Add the project demonstration video here:

```text
Coming Soon
```

---

# 📈 MLflow Experiment Tracking

MLflow is included for experiment tracking and model development.

Tracked information can include:

* Accuracy
* ROC-AUC
* Failure Recall
* Model artifacts
* Experiment parameters
* Experiment results

The repository includes:

```text
mlruns/
```

for MLflow experiment information.

---

# 🛠️ Tech Stack

| Category             | Technologies                   |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| Data Processing      | Pandas, NumPy                  |
| Visualization        | Matplotlib, Seaborn            |
| Machine Learning     | Scikit-learn                   |
| ML Model             | Random Forest                  |
| Deep Learning        | TensorFlow / Keras             |
| Explainability       | SHAP / Feature Importance      |
| LLM                  | Groq                           |
| RAG                  | Retrieval-Augmented Generation |
| Computer Vision      | Pillow / Image Analysis        |
| Multi-Agent AI       | Python-based Agents            |
| Simulation           | Digital Twin                   |
| MLOps                | MLflow                         |
| Dashboard            | Streamlit                      |
| Environment          | python-dotenv                  |
| Version Control      | Git & GitHub                   |

---

# 📁 Project Structure

```text
factory-ai-predictive-maintenance/
│
├── .devcontainer/
│
├── app/
│   └── streamlit_app.py
│
├── agents/
│   ├── predictive_agent.py
│   ├── vision_agent.py
│   ├── knowledge_agent.py
│   └── planning_agent.py
│
├── data/
│   ├── factory_data.csv
│   └── images/
│
├── deep_learning/
│   └── model/
│       ├── factory_failure_mlp.keras
│       └── preprocessor.pkl
│
├── models/
│
├── rag/
│
├── xai/
│
├── reports/
│
├── mlruns/
│
├── config.py
├── digital_twin.py
├── main.py
├── mlflow_tracking.py
├── mlflow_train.py
├── requirements.txt
├── .gitignore
└── README.md
```

> Project structure may evolve as additional components are added.

---

# ⚡ Installation

## 1. Clone Repository

```bash
git clone https://github.com/rooj908/factory-ai-predictive-maintenance.git
```

## 2. Open Project

```bash
cd factory-ai-predictive-maintenance
```

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit API keys or `.env` files to GitHub.

Make sure `.env` is included in `.gitignore`.

---

# ▶️ Run the Project

## Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Main Predictive Maintenance System

```bash
python main.py
```

---

## MLflow Training

```bash
python mlflow_train.py
```

---

## Digital Twin

```bash
python digital_twin.py
```

---

# 📦 Requirements

The project uses:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
python-dotenv
groq
Pillow
shap
mlflow
tensorflow==2.21.0
joblib
```

---

# 🔮 Future Improvements

Future versions of Factory AI can include:

* 🌐 Real-time IoT sensor integration
* 📡 Live sensor streaming
* 👁️ Advanced computer vision defect detection
* 🚨 Advanced anomaly detection
* ☁️ Cloud deployment
* 📅 Automated maintenance scheduling
* 🏭 SCADA / PLC integration
* 📊 Real-time machine monitoring
* 🧠 Advanced predictive models
* 📝 Automated maintenance history
* 🔄 Continuous model retraining
* 📱 Mobile maintenance alerts

---

# 🔒 Safety & Responsible Use

Factory AI is an AI-assisted decision-support system.

It should **not independently control industrial machinery**.

Actual maintenance decisions should be reviewed by qualified personnel and carried out according to appropriate industrial safety procedures.

Predictions and simulated costs should be treated as decision-support information rather than guaranteed outcomes.

---

# 👩‍💻 Team

## Urooj Fatima

Data Science / AI

GitHub:
https://github.com/rooj908

---

## Unsha Siddiqui

Data Science / AI

GitHub:
https://github.com/unshaimran

---

## 🤝 Project Collaboration

**Factory AI was collaboratively developed by Urooj Fatima and Unsha Siddiqui.**

The project combines our work across:

* Data Science
* Machine Learning
* Deep Learning
* Explainable AI
* RAG
* Computer Vision
* AI Agents
* Digital Twin
* Streamlit

---

# 🌟 Key Highlights

```text
╔══════════════════════════════════════════════╗
║                 FACTORY AI                   ║
║      Intelligent Predictive Maintenance      ║
╠══════════════════════════════════════════════╣
║                                              ║
║  ✓ Machine Failure Prediction                ║
║  ✓ Random Forest Classification              ║
║  ✓ Deep Learning MLP                         ║
║  ✓ ~98% ML Accuracy                          ║
║  ✓ ~0.997 ML ROC-AUC                         ║
║  ✓ Explainable AI                            ║
║  ✓ RAG Maintenance Knowledge                 ║
║  ✓ Multi-Agent AI Workflow                   ║
║  ✓ Computer Vision Support                   ║
║  ✓ Digital Twin Simulation                   ║
║  ✓ Human-in-the-Loop Decision Making         ║
║  ✓ MLflow Experiment Tracking                ║
║  ✓ Interactive Streamlit Dashboard           ║
║                                              ║
╚══════════════════════════════════════════════╝
```

---

# 💡 Project Vision

Factory AI aims to demonstrate how modern AI systems can help industries move from:

```text
Reactive Maintenance
        ↓
Scheduled Maintenance
        ↓
Predictive Maintenance
        ↓
Intelligent Decision Support
```

The ultimate goal is to help maintenance teams:

* Identify machine risks earlier
* Understand why a machine may be at risk
* Retrieve relevant maintenance knowledge
* Inspect machines visually
* Compare possible maintenance strategies
* Estimate potential costs and downtime
* Make better-informed maintenance decisions

---

# 📌 Hackathon Summary

**Project:** Factory AI: Intelligent Predictive Maintenance & Decision Support System

**Developed by:** Urooj Fatima & Unsha Siddiqui

**Repository:**
https://github.com/rooj908/factory-ai-predictive-maintenance

**GitHub:**
https://github.com/rooj908

**Collaborator:**
https://github.com/unshaimran

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

**Built with Python, Machine Learning, Deep Learning, RAG, AI Agents, Computer Vision, MLflow, Digital Twin, and Streamlit.**
