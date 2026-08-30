import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(
    page_title="Factory AI",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Factory AI Decision Support System")
st.caption("Predictive Maintenance + Computer Vision + RAG + Multi-Agent AI")

# =========================================================
# 1. PREDICTIVE MAINTENANCE
# =========================================================

st.header("1. Predictive Maintenance")

failure_probability = st.slider(
    "Failure Probability",
    0.0,
    1.0,
    0.78,
    0.01
)

if failure_probability >= 0.70:
    risk = "HIGH"
elif failure_probability >= 0.40:
    risk = "MEDIUM"
else:
    risk = "LOW"

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Failure Probability",
        f"{failure_probability * 100:.1f}%"
    )

with col2:
    st.metric(
        "Risk Level",
        risk
    )

st.divider()

# =========================================================
# 2. COMPUTER VISION
# =========================================================

st.header("2. Computer Vision")

uploaded_image = st.file_uploader(
    "📷 Upload Machine Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded Machine Image",
        use_container_width=True
    )

    image_dir = Path("data/images")
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / uploaded_image.name

    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    vision_result = {
        "defect": "Mechanical anomaly",
        "severity": "Medium",
        "confidence": 0.85
    }

    st.success("Image received by Vision Agent")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Detected Defect",
            vision_result["defect"]
        )

    with col2:
        st.metric(
            "Severity",
            vision_result["severity"]
        )

    with col3:
        st.metric(
            "Confidence",
            "85%"
        )

else:

    st.info(
        "Please upload a machine image for visual inspection."
    )

st.divider()

# =========================================================
# 3. EXPLAINABLE AI
# =========================================================

st.header("3. Explainable AI")

features = {
    "Temperature": 0.238,
    "RPM": 0.215,
    "Current": 0.169,
    "Vibration": 0.155,
    "Hours Since Maintenance": 0.077,
    "Pressure": 0.060
}

xai_df = pd.DataFrame(
    list(features.items()),
    columns=["Feature", "Importance"]
)

st.bar_chart(
    xai_df.set_index("Feature")
)

st.info(
    "Temperature, RPM, current and vibration are the "
    "most influential machine-health signals."
)

st.divider()

# =========================================================
# 4. RAG KNOWLEDGE BASE
# =========================================================

st.header("4. RAG Knowledge Evidence")

with st.expander("High Vibration Maintenance Guide"):

    st.write(
        "High vibration may indicate bearing wear, "
        "shaft misalignment, imbalance or mechanical looseness."
    )

    st.write(
        "Inspect bearings, coupling and shaft alignment."
    )

    st.write(
        "If vibration continues after inspection, schedule "
        "maintenance before returning the machine to full production."
    )

with st.expander("Preventive Maintenance SOP"):

    st.write(
        "Machines should be inspected regularly based on "
        "operating hours, sensor trends and maintenance history."
    )

    st.write(
        "Record maintenance actions and monitor vibration, "
        "temperature, pressure and current after servicing."
    )

with st.expander("Electrical Safety SOP"):

    st.write(
        "Before inspecting electrical components, isolate "
        "the machine from the power source and follow "
        "lockout/tagout procedures."
    )

    st.write(
        "Only authorized personnel should perform electrical maintenance."
    )

st.divider()

# =========================================================
# 5. MULTI-AGENT WORKFLOW
# =========================================================

st.header("5. Multi-Agent Workflow")

agents = {
    "Predictive Maintenance Agent":
        "Failure probability: 78% | Risk: HIGH",

    "Vision Agent":
        "Mechanical anomaly | Severity: Medium | Confidence: 85%",

    "Knowledge Agent":
        "Retrieved High Vibration Maintenance Guide and SOP evidence",

    "Planning / Decision Agent":
        "Inspect machine before returning it to full production"
}

for agent, output in agents.items():

    with st.expander(agent):
        st.write(output)

st.divider()

# =========================================================
# 6. DIGITAL TWIN
# =========================================================

st.header("6. Digital Twin What-If Simulation")

simulation = pd.DataFrame({

    "Scenario": [
        "Continue Operation",
        "Stop for Maintenance",
        "Reduce Machine Load"
    ],

    "Downtime Hours": [
        0,
        4,
        1
    ],

    "Failure Risk": [
        0.78,
        0.15,
        0.35
    ],

    "Production Loss": [
        0,
        400,
        100
    ],

    "Estimated Cost": [
        3900,
        8750,
        3750
    ]
})

st.dataframe(
    simulation,
    use_container_width=True
)

st.success(
    "Recommended Scenario: Reduce Machine Load | "
    "Estimated Cost: 3750"
)

st.divider()

# =========================================================
# 7. HUMAN-IN-THE-LOOP
# =========================================================

st.header("7. Human Supervisor Decision")

st.write("AI Recommendation:")

st.info(
    "Inspect machine before returning it to full production."
)

decision = st.radio(
    "Supervisor Decision",
    ["APPROVE", "REJECT", "MODIFY"]
)

reason = st.text_area(
    "Reason / Modification"
)

if st.button("Record Human Decision"):

    record = {
        "ai_recommendation":
            "Inspect machine before returning it to full production.",

        "human_decision":
            decision,

        "reason":
            reason
    }

    Path("reports").mkdir(exist_ok=True)

    with open(
        "reports/web_decision.json",
        "w"
    ) as f:

        json.dump(
            record,
            f,
            indent=4
        )

    st.success(
        f"Human decision recorded: {decision}"
    )

st.divider()

# =========================================================
# 8. FINAL RECOMMENDATION
# =========================================================

st.header("8. Final AI Recommendation")

if risk == "HIGH":

    st.warning(
        "HIGH RISK: Inspect the machine before returning "
        "it to full production."
    )

elif risk == "MEDIUM":

    st.warning(
        "MEDIUM RISK: Reduce machine load and perform "
        "preventive inspection."
    )

else:

    st.success(
        "LOW RISK: Continue operation with routine monitoring."
    )

st.caption(
    "AI output is decision support. Final operational "
    "authority remains with the human supervisor."
)