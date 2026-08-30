# 🏭 Factory AI Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)](https://mlflow.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/rooj908/factory-ai-predictive-maintenance)

> **AI-powered industrial predictive maintenance and decision-support system**

Factory AI is an intelligent predictive maintenance system that analyzes industrial machine data to detect failure risk, explain predictions, retrieve relevant maintenance knowledge, evaluate machine images, simulate operational scenarios, and support human maintenance decisions.

---

## 🏆 Hackathon Project

### Factory AI: Intelligent Predictive Maintenance & Decision Support System

This project was developed as a hackathon solution for applying **Artificial Intelligence and Data Science to industrial maintenance**.

The system combines:

**Machine Learning + Explainable AI + RAG + AI Agents + Computer Vision + Digital Twin + MLflow + Streamlit**

to create an integrated predictive maintenance workflow.

---

# 🎯 Problem Statement

Unexpected industrial machine failures can result in:

* Production downtime
* Maintenance costs
* Equipment damage
* Reduced productivity
* Safety risks

Traditional maintenance approaches are often reactive or based on fixed schedules.

### Our Solution

Factory AI uses machine sensor data and AI techniques to identify potential failure risk **before a breakdown occurs** and provide evidence-based maintenance recommendations.

The system helps answer:

> **"Is this machine at risk, why is it at risk, and what should we do next?"**

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │   Machine Sensors    │
                    │ Temperature / RPM    │
                    │ Vibration / Pressure  │
                    │ Current / Maintenance │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Analysis & EDA  │
                    │ Cleaning / Analysis  │
                    │ Feature Investigation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Predictive ML Model  │
                    │   Random Forest      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Failure Risk         │
                    │ LOW / MEDIUM / HIGH  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │    XAI     │   │    RAG     │   │   Vision   │
       │ Explain    │   │ Knowledge  │   │ Inspection │
       │ Prediction │   │ Retrieval  │   │            │
       └──────┬─────┘   └──────┬─────┘   └──────┬─────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   AI Agent Workflow  │
                    │ Analysis & Planning  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Digital Twin      │
                    │    What-if Analysis  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Human Supervisor     │
                    │ APPROVE / REJECT /   │
                    │ MODIFY               │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Final Maintenance    │
                    │ Decision             │
                    └──────────────────────┘
```

---

# ⚙️ How It Works

### Step 1: Machine Data

The system receives industrial sensor and operational data including:

* Motor temperature
* RPM
* Vibration RMS
* Average phase current
* Pressure level
* Hours since maintenance
* Ambient temperature

### Step 2: Data Analysis

The data is analyzed through:

* Dataset inspection
* Data type analysis
* Missing-value analysis
* Statistical analysis
* Distribution analysis
* Outlier investigation
* Correlation analysis
* Feature relationships
* Failure-related analysis

### Step 3: Predictive Model

A Random Forest classification model predicts the probability of machine failure.

### Step 4: Risk Assessment

The predicted failure probability is converted into an operational risk level:

```text
LOW
MEDIUM
HIGH
```

### Step 5: Explainability

XAI techniques identify the important factors contributing to the prediction.

### Step 6: Knowledge Retrieval

The RAG component retrieves relevant maintenance procedures and safety information.

### Step 7: Visual Inspection

Machine images can be uploaded for visual inspection as part of the maintenance assessment.

### Step 8: AI Decision Support

The AI workflow combines prediction, visual information, and maintenance knowledge to produce a recommendation.

### Step 9: Digital Twin

Different operational scenarios can be compared to estimate:

* Failure risk
* Downtime
* Production impact
* Estimated cost

### Step 10: Human Approval

A human supervisor reviews the recommendation before the final maintenance decision.

---

# 🤖 AI Components

## 🔮 Predictive Maintenance Agent

Analyzes machine sensor data and provides:

* Failure probability
* Risk classification
* Maintenance recommendation

---

## 👁️ Vision Agent

Processes uploaded machine images and provides a visual inspection result.

---

## 📚 Knowledge Agent

Uses Retrieval-Augmented Generation to retrieve relevant maintenance knowledge from the project's knowledge base.

---

## 🧠 Planning / Decision Agent

Combines information from:

* Machine learning
* XAI
* Vision
* RAG
* Digital Twin

to generate a final recommendation for human review.

---

# 📊 Machine Learning Performance

The current predictive model achieved:

| Metric         | Result |
| -------------- | -----: |
| Accuracy       |   ~98% |
| ROC-AUC        | ~0.997 |
| Failure Recall |  ~0.96 |
| Test Samples   |  4,809 |

> Results can vary when the model is retrained with different data splits or configurations.

---

# ⭐ Important Features

The major predictive features include:

1. **Motor Temperature**
2. **RPM**
3. **Average Phase Current**
4. **Vibration RMS**
5. **Hours Since Maintenance**
6. **Pressure Level**
7. **Ambient Temperature**

---

# 🔍 Explainable AI

Factory AI does not treat the model as a black box.

The XAI component helps identify which features contribute to machine failure predictions.

This provides maintenance teams with additional context instead of showing only:

```text
Failure = YES
```

The system can instead provide information about **why the machine is considered risky**.

---

# 📚 RAG Maintenance Knowledge

The RAG knowledge base contains maintenance-related information such as:

* High Vibration Maintenance Guide
* Preventive Maintenance SOP
* Electrical Safety SOP

The Knowledge Agent retrieves relevant information based on the machine condition.

### Example

```text
Machine condition:
High vibration + elevated temperature

↓

Retrieve relevant maintenance information

↓

Recommendation:
Inspect machine before returning it to full production.
```

---

# 🪞 Digital Twin Simulation

The Digital Twin component allows the user to compare different operational scenarios.

### Example

| Scenario             | Downtime | Failure Risk | Estimated Cost |
| -------------------- | -------: | -----------: | -------------: |
| Continue Operation   |       0h |          78% |           3900 |
| Stop for Maintenance |       4h |          15% |           8750 |
| Reduce Machine Load  |       1h |          35% |           3750 |

This allows a supervisor to compare possible decisions instead of relying only on a single model prediction.

---

# 👨‍💼 Human-in-the-Loop

Factory AI is designed as a **decision-support system**.

The AI does not independently control industrial equipment.

A human supervisor can:

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

The final operational decision remains under human supervision.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard that brings the major components together.

### Dashboard Features

* 📊 Machine condition analysis
* 🤖 Failure-risk prediction
* 🚦 Risk classification
* 👁️ Machine image upload
* 🔍 Explainable AI insights
* 📚 Maintenance knowledge retrieval
* 🧠 AI-generated recommendations
* 🪞 Digital Twin what-if analysis
* 👨‍💼 Human decision support

### Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

---

# 📸 Dashboard Preview

> Add screenshots of the Streamlit dashboard here after taking them during the demo.

```text
docs/
└── screenshots/
    ├── dashboard.png
    ├── prediction.png
    ├── xai.png
    ├── digital_twin.png
    └── final_decision.png
```

Example Markdown:

```markdown
![Factory AI Dashboard](docs/screenshots/dashboard.png)
```

---

# 🎥 Demo

### Live Demo

Add your deployed Streamlit URL here:

```text
Coming Soon
```

### Demo Video

Add your YouTube or Google Drive demo link here:

```text
Coming Soon
```

---

# 📈 MLflow Experiment Tracking

MLflow is used to track model experiments and artifacts.

Tracked information includes:

* Accuracy
* ROC-AUC
* Failure Recall
* Model artifacts
* Experiment information

The repository contains the MLflow experiment directory:

```text
mlruns/
```

---

# 🛠️ Tech Stack

| Category         | Technologies                   |
| ---------------- | ------------------------------ |
| Language         | Python                         |
| Data Processing  | Pandas, NumPy                  |
| Visualization    | Matplotlib, Seaborn            |
| Machine Learning | Scikit-learn                   |
| Model            | Random Forest                  |
| Explainability   | XAI / Feature Importance       |
| LLM              | Groq                           |
| RAG              | Retrieval-Augmented Generation |
| Computer Vision  | Image-based inspection         |
| Simulation       | Digital Twin                   |
| MLOps            | MLflow                         |
| Dashboard        | Streamlit                      |
| Version Control  | Git & GitHub                   |

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
├── data/
│   └── factory_data.csv
│
├── models/
│
├── rag/
│
├── reports/
│
├── xai/
│
├── mlruns/
│   └── MLflow artifacts
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

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit API keys or `.env` files to GitHub.

---

# ▶️ Run the Project

### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Main Predictive Maintenance System

```bash
python main.py
```

### MLflow Training

```bash
python mlflow_train.py
```

### Digital Twin

```bash
python digital_twin.py
```

---

# 🔮 Future Improvements

Future versions can include:

* 🌐 Real-time IoT sensor integration
* 📡 Live sensor streaming
* 👁️ Advanced computer vision defect detection
* 🚨 Advanced anomaly detection
* ☁️ Cloud deployment
* 📅 Automated maintenance scheduling
* 🏭 SCADA / PLC integration
* 📊 Real-time machine monitoring
* 🧠 More advanced predictive models
* 📝 Automated maintenance history
* 🔄 Continuous model retraining

---

# 🔒 Safety & Responsible Use

Factory AI is an AI-assisted decision-support system.

It should not independently control industrial machinery.

Actual maintenance operations should always be reviewed by qualified personnel and performed according to appropriate industrial safety procedures.

---

# 👩‍💻 Team

## Urooj Fatima

GitHub:
https://github.com/rooj908

## Unsha Siddiqui

GitHub:
https://github.com/unshaimran

### Project Collaboration

**Factory AI was collaboratively developed by Urooj Fatima and Unsha Siddiqui.**

---

# 🌟 Key Highlights

```text
╔══════════════════════════════════════════════╗
║             FACTORY AI                       ║
║     Intelligent Predictive Maintenance       ║
╠══════════════════════════════════════════════╣
║                                              ║
║  ✓ Machine Failure Prediction                ║
║  ✓ Random Forest Classification              ║
║  ✓ ~98% Model Accuracy                       ║
║  ✓ ~0.997 ROC-AUC                             ║
║  ✓ Explainable AI                            ║
║  ✓ RAG Maintenance Knowledge                 ║
║  ✓ AI Agent Workflow                         ║
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

The ultimate goal is to help maintenance teams identify risks earlier, understand the reasons behind predictions, compare possible actions, and make better-informed decisions.

---

## 📌 Hackathon

**Project:** Factory AI: Intelligent Predictive Maintenance & Decision Support System

**Developed by:**
**Urooj Fatima & Unsha Siddiqui**

**Repository:**
https://github.com/rooj908/factory-ai-predictive-maintenance
