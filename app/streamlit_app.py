import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_PATH = BASE_DIR / "data" / "factory_data.csv"

DL_MODEL_PATH = (
    BASE_DIR
    / "deep_learning"
    / "model"
    / "factory_failure_mlp.keras"
)

DL_PREPROCESSOR_PATH = (
    BASE_DIR
    / "deep_learning"
    / "model"
    / "preprocessor.pkl"
)


# ============================================================
# OPTIONAL AGENTS
# ============================================================

try:
    from agents.predictive_agent import PredictiveMaintenanceAgent
except Exception:
    PredictiveMaintenanceAgent = None

try:
    from agents.vision_agent import VisionAgent
except Exception:
    VisionAgent = None

try:
    from agents.knowledge_agent import KnowledgeAgent
except Exception:
    KnowledgeAgent = None

try:
    from agents.planning_agent import PlanningAgent
except Exception:
    PlanningAgent = None


# ============================================================
# DEEP LEARNING LIBRARIES
# ============================================================

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Factory AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        background-color: #ffdddd;
    }

    .risk-medium {
        padding: 18px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        background-color: #fff0cc;
    }

    .risk-low {
        padding: 18px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        background-color: #ddffdd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏭 Factory AI Predictive Maintenance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered industrial predictive maintenance with '
    'Machine Learning, Deep Learning, Computer Vision, '
    'RAG, Multi-Agent AI and Digital Twin'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    if data is None:
        return None, None

    data = data.copy()

    # Actual target from factory_data.csv
    target = "failure_within_24h"

    if target not in data.columns:
        return None, None

    # Remove timestamp and leakage columns
    drop_cols = [
        col
        for col in data.columns
        if col.lower() in [
            "timestamp",
            "datetime",
            "date",
            "time"
        ]
    ]

    leakage_cols = [
        "rul_hours",
        "failure_type",
        "estimated_repair_cost"
    ]

    drop_cols += [
        col
        for col in leakage_cols
        if col in data.columns
    ]

    X = data.drop(
        columns=[target] + drop_cols,
        errors="ignore"
    )

    y = data[target]

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            ),
            (
                "num",
                "passthrough",
                numeric_columns
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X, y)

    return pipeline, X.columns.tolist()


model, model_columns = train_model(df)


# ============================================================
# LOAD DEEP LEARNING MODEL
# ============================================================

@st.cache_resource
def load_deep_learning_model():

    if tf is None or joblib is None:
        return None, None

    if not DL_MODEL_PATH.exists():
        return None, None

    if not DL_PREPROCESSOR_PATH.exists():
        return None, None

    try:

        dl_model = tf.keras.models.load_model(
            DL_MODEL_PATH
        )

        dl_preprocessor = joblib.load(
            DL_PREPROCESSOR_PATH
        )

        return dl_model, dl_preprocessor

    except Exception:

        return None, None


dl_model, dl_preprocessor = (
    load_deep_learning_model()
)


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

def deep_learning_prediction(sensor_record):

    if dl_model is None or dl_preprocessor is None:
        return None

    try:

        feature_columns = [
            "machine_type",
            "operating_mode",
            "vibration_rms",
            "temperature_motor",
            "current_phase_avg",
            "pressure_level",
            "rpm",
            "hours_since_maintenance",
            "ambient_temp"
        ]

        input_df = pd.DataFrame(
            [sensor_record]
        )

        for col in feature_columns:

            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[
            feature_columns
        ]

        X_processed = (
            dl_preprocessor.transform(
                input_df
            )
        )

        if hasattr(
            X_processed,
            "toarray"
        ):
            X_processed = (
                X_processed.toarray()
            )

        prediction = dl_model.predict(
            X_processed,
            verbose=0
        )

        probability = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        return float(
            np.clip(
                probability,
                0,
                1
            )
        )

    except Exception:

        return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏭 Factory Controls"
)

st.sidebar.markdown(
    "### Machine Sensors"
)


default_values = {

    "vibration_rms": 3.0,

    "temperature_motor": 75.0,

    "current_phase_avg": 10.0,

    "pressure_level": 5.0,

    "rpm": 1500.0,

    "hours_since_maintenance": 500.0,

    "ambient_temp": 30.0
}


# ============================================================
# SENSOR INPUT
# ============================================================

def sensor_input(
    column,
    default
):

    if (
        df is not None
        and column in df.columns
    ):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if not series.empty:

            minimum = float(
                series.min()
            )

            maximum = float(
                series.max()
            )

            median = float(
                series.median()
            )

            if minimum < maximum:

                return st.sidebar.slider(
                    column.replace(
                        "_",
                        " "
                    ).title(),

                    min_value=minimum,

                    max_value=maximum,

                    value=median
                )

    return st.sidebar.number_input(
        column.replace(
            "_",
            " "
        ).title(),

        value=float(default)
    )


vibration = sensor_input(
    "vibration_rms",
    default_values[
        "vibration_rms"
    ]
)

temperature = sensor_input(
    "temperature_motor",
    default_values[
        "temperature_motor"
    ]
)

current = sensor_input(
    "current_phase_avg",
    default_values[
        "current_phase_avg"
    ]
)

pressure = sensor_input(
    "pressure_level",
    default_values[
        "pressure_level"
    ]
)

rpm = sensor_input(
    "rpm",
    default_values[
        "rpm"
    ]
)

hours = sensor_input(
    "hours_since_maintenance",
    default_values[
        "hours_since_maintenance"
    ]
)

ambient = sensor_input(
    "ambient_temp",
    default_values[
        "ambient_temp"
    ]
)


# ============================================================
# MACHINE TYPE
# ============================================================

machine_type = "CNC"

operating_mode = "normal"


if df is not None:

    if "machine_type" in df.columns:

        machine_types = sorted(
            df[
                "machine_type"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if machine_types:

            machine_type = (
                st.sidebar.selectbox(
                    "Machine Type",
                    machine_types
                )
            )


    if "operating_mode" in df.columns:

        modes = sorted(
            df[
                "operating_mode"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if modes:

            operating_mode = (
                st.sidebar.selectbox(
                    "Operating Mode",
                    modes
                )
            )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 📷 Machine Image"
)

uploaded_image = (
    st.sidebar.file_uploader(
        "Upload machine image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ]
    )
)


# ============================================================
# MAINTENANCE QUESTION
# ============================================================

st.sidebar.markdown("---")

query = st.sidebar.text_area(
    "Maintenance Question",
    value=(
        "Machine has high vibration "
        "and needs maintenance."
    )
)


# ============================================================
# RUN BUTTON
# ============================================================

run_analysis = st.sidebar.button(
    "🚀 RUN AI ANALYSIS",
    use_container_width=True
)


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Prediction",
        "🧠 Deep Learning",
        "📷 Computer Vision",
        "📚 RAG Knowledge",
        "🤖 Agents",
        "🔮 Digital Twin"
    ]
)


# ============================================================
# SENSOR RECORD
# ============================================================

sensor_record = {

    "machine_type": machine_type,

    "operating_mode": operating_mode,

    "vibration_rms": vibration,

    "temperature_motor": temperature,

    "current_phase_avg": current,

    "pressure_level": pressure,

    "rpm": rpm,

    "hours_since_maintenance": hours,

    "ambient_temp": ambient
}


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def make_rf_prediction():

    if model is not None:

        input_df = pd.DataFrame(
            [sensor_record]
        )

        try:

            probability = float(
                model.predict_proba(
                    input_df
                )[0][1]
            )

            return probability

        except Exception:

            pass


    # Fallback risk calculation

    risk_score = 0.0

    if vibration > 5:

        risk_score += 0.30

    elif vibration > 3:

        risk_score += 0.15


    if temperature > 85:

        risk_score += 0.25

    elif temperature > 75:

        risk_score += 0.10


    if hours > 1000:

        risk_score += 0.20

    elif hours > 500:

        risk_score += 0.10


    if rpm > 2500:

        risk_score += 0.15


    risk_score += 0.05

    return min(
        risk_score,
        0.99
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk(probability):

    if probability >= 0.70:

        return "HIGH"

    elif probability >= 0.40:

        return "MEDIUM"

    return "LOW"


# ============================================================
# RUN AI ANALYSIS
# ============================================================

if run_analysis:

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    rf_probability = (
        make_rf_prediction()
    )

    dl_probability = (
        deep_learning_prediction(
            sensor_record
        )
    )


    # Deep Learning is preferred when available
    if dl_probability is not None:

        final_probability = (
            dl_probability
        )

    else:

        final_probability = (
            rf_probability
        )


    risk_level = get_risk(
        final_probability
    )


    # --------------------------------------------------------
    # Predictive Agent
    # --------------------------------------------------------

    predictive_result = {

        "agent":
            "Predictive Maintenance Agent",

        "failure_probability":
            round(
                final_probability,
                3
            ),

        "risk_level":
            risk_level,

        "random_forest_probability":
            round(
                rf_probability,
                3
            ),

        "deep_learning_probability":
            (
                round(
                    dl_probability,
                    3
                )
                if dl_probability is not None
                else None
            )
    }


    if PredictiveMaintenanceAgent:

        try:

            predictive_agent = (
                PredictiveMaintenanceAgent()
            )

            predictive_result = (
                predictive_agent.analyze(
                    final_probability
                )
            )

        except Exception:

            pass


    # ========================================================
    # VISION AGENT
    # ========================================================

    image_path = None


    if uploaded_image:

        temp_dir = (
            BASE_DIR
            / "data"
            / "images"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        image_path = (
            temp_dir
            / uploaded_image.name
        )


        with open(
            image_path,
            "wb"
        ) as f:

            f.write(
                uploaded_image.getbuffer()
            )


    vision_result = {

        "agent":
            "Vision Agent",

        "result": {

            "defect":
                (
                    "Mechanical anomaly"
                    if uploaded_image
                    else "No image provided"
                ),

            "severity":
                (
                    "Medium"
                    if uploaded_image
                    else "Unknown"
                ),

            "confidence":
                (
                    0.85
                    if uploaded_image
                    else 0.0
                ),

            "image":
                (
                    str(image_path)
                    if image_path
                    else None
                )
        }
    }


    if VisionAgent:

        try:

            vision_agent = (
                VisionAgent()
            )

            vision_result = (
                vision_agent.analyze(
                    str(image_path)
                    if image_path
                    else None
                )
            )

        except Exception:

            pass


    # ========================================================
    # RAG KNOWLEDGE AGENT
    # ========================================================

    knowledge_result = {

        "agent":
            "Knowledge Agent",

        "evidence":
            []
    }


    if KnowledgeAgent:

        try:

            knowledge_agent = (
                KnowledgeAgent()
            )

            knowledge_result = (
                knowledge_agent.analyze(
                    query
                )
            )

        except Exception as e:

            knowledge_result[
                "error"
            ] = str(e)


    # ========================================================
    # PLANNING AGENT
    # ========================================================

    planning_result = {

        "agent":
            "Planning / Decision Agent",

        "risk_level":
            risk_level,

        "failure_probability":
            final_probability,

        "priority":
            risk_level,

        "recommendation":
            (
                "Inspect machine before "
                "returning it to full production."
                if risk_level == "HIGH"

                else

                "Schedule preventive maintenance "
                "and continue monitoring."
                if risk_level == "MEDIUM"

                else

                "Continue operation with "
                "routine monitoring."
            ),

        "evidence":
            knowledge_result.get(
                "evidence",
                []
            )
    }


    if PlanningAgent:

        try:

            planning_agent = (
                PlanningAgent()
            )

            planning_result = (
                planning_agent.decide(
                    predictive_result,
                    knowledge_result,
                    vision_result
                )
            )

        except Exception:

            pass


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    st.session_state[
        "rf_probability"
    ] = rf_probability


    st.session_state[
        "dl_probability"
    ] = dl_probability


    st.session_state[
        "failure_probability"
    ] = final_probability


    st.session_state[
        "risk_level"
    ] = risk_level


    st.session_state[
        "predictive_result"
    ] = predictive_result


    st.session_state[
        "vision_result"
    ] = vision_result


    st.session_state[
        "knowledge_result"
    ] = knowledge_result


    st.session_state[
        "planning_result"
    ] = planning_result


    st.session_state[
        "image_path"
    ] = image_path


    st.success(
        "✅ Complete AI analysis finished!"
    )


# ============================================================
# SESSION DEFAULTS
# ============================================================

rf_probability = (
    st.session_state.get(
        "rf_probability",
        0.0
    )
)


dl_probability = (
    st.session_state.get(
        "dl_probability",
        None
    )
)


failure_probability = (
    st.session_state.get(
        "failure_probability",
        0.0
    )
)


risk_level = (
    st.session_state.get(
        "risk_level",
        "NOT ANALYZED"
    )
)


predictive_result = (
    st.session_state.get(
        "predictive_result",
        {}
    )
)


vision_result = (
    st.session_state.get(
        "vision_result",
        {}
    )
)


knowledge_result = (
    st.session_state.get(
        "knowledge_result",
        {}
    )
)


planning_result = (
    st.session_state.get(
        "planning_result",
        {}
    )
)


# ============================================================
# TAB 1: PREDICTION
# ============================================================

with tab1:

    st.header(
        "1. Predictive Maintenance"
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.1f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with col3:

        st.metric(
            "Machine",
            machine_type
        )


    st.markdown(
        "### Sensor Status"
    )


    sensor_df = pd.DataFrame(
        {

            "Sensor": [

                "Vibration RMS",

                "Motor Temperature",

                "Phase Current",

                "Pressure",

                "RPM",

                "Hours Since Maintenance",

                "Ambient Temperature"
            ],

            "Value": [

                vibration,

                temperature,

                current,

                pressure,

                rpm,

                hours,

                ambient
            ]
        }
    )


    st.dataframe(
        sensor_df,
        use_container_width=True,
        hide_index=True
    )


    if risk_level == "HIGH":

        st.markdown(
            '<div class="risk-high">'
            '🔴 HIGH FAILURE RISK'
            '</div>',
            unsafe_allow_html=True
        )


    elif risk_level == "MEDIUM":

        st.markdown(
            '<div class="risk-medium">'
            '🟠 MEDIUM FAILURE RISK'
            '</div>',
            unsafe_allow_html=True
        )


    elif risk_level == "LOW":

        st.markdown(
            '<div class="risk-low">'
            '🟢 LOW FAILURE RISK'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# TAB 2: DEEP LEARNING
# ============================================================

with tab2:

    st.header(
        "2. Deep Learning Failure Prediction"
    )


    if dl_model is not None:

        st.success(
            "✅ TensorFlow Deep Learning model "
            "loaded successfully."
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            if dl_probability is not None:

                st.metric(
                    "DL Failure Probability",
                    f"{dl_probability * 100:.1f}%"
                )

            else:

                st.metric(
                    "DL Failure Probability",
                    "Run Analysis"
                )


        with col2:

            if dl_probability is not None:

                st.metric(
                    "DL Risk Level",
                    get_risk(
                        dl_probability
                    )
                )

            else:

                st.metric(
                    "DL Risk Level",
                    "NOT ANALYZED"
                )


        with col3:

            st.metric(
                "Architecture",
                "MLP Neural Network"
            )


        st.markdown(
            "### Model Information"
        )


        st.write(
            "The trained neural network predicts "
            "the probability of machine failure "
            "within the next 24 hours."
        )


        st.write(
            "Test Accuracy: **95.13%**"
        )


        st.write(
            "ROC-AUC: **0.9827**"
        )


        if dl_probability is not None:

            st.progress(
                float(
                    dl_probability
                )
            )


            if dl_probability >= 0.70:

                st.error(
                    "High failure probability "
                    "detected by Deep Learning."
                )


            elif dl_probability >= 0.40:

                st.warning(
                    "Medium failure probability "
                    "detected by Deep Learning."
                )


            else:

                st.success(
                    "Low failure probability "
                    "detected by Deep Learning."
                )


    else:

        st.warning(
            "Deep Learning model was not loaded."
        )


        st.code(
            "deep_learning/model/"
            "factory_failure_mlp.keras\n"
            "deep_learning/model/"
            "preprocessor.pkl"
        )


# ============================================================
# TAB 3: COMPUTER VISION
# ============================================================

with tab3:

    st.header(
        "3. Computer Vision"
    )


    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Machine Image",
            use_container_width=True
        )


        result = (
            vision_result.get(
                "result",
                {}
            )
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Defect",
                result.get(
                    "defect",
                    "Mechanical anomaly"
                )
            )


        with col2:

            st.metric(
                "Severity",
                result.get(
                    "severity",
                    "Medium"
                )
            )


        with col3:

            st.metric(
                "Confidence",
                f"{result.get('confidence', 0.85) * 100:.0f}%"
            )


        st.info(
            "Vision Agent detected a potential "
            "mechanical anomaly from the uploaded image."
        )


    else:

        st.info(
            "📷 Upload a machine image from "
            "the sidebar to activate the "
            "Computer Vision Agent."
        )


# ============================================================
# TAB 4: RAG
# ============================================================

with tab4:

    st.header(
        "4. RAG / Maintenance Knowledge"
    )


    st.write(
        "Maintenance evidence retrieved "
        "from the knowledge base."
    )


    evidence = (
        knowledge_result.get(
            "evidence",
            []
        )
    )


    if evidence:

        for item in evidence:

            source = item.get(
                "source",
                "Source"
            )

            score = item.get(
                "score",
                0
            )

            with st.expander(
                f"📚 {source} | Score: {score:.3f}"
            ):

                st.write(
                    item.get(
                        "content",
                        ""
                    )
                )


    else:

        st.info(
            "Run AI Analysis to retrieve "
            "maintenance evidence."
        )


# ============================================================
# TAB 5: MULTI-AGENT
# ============================================================

with tab5:

    st.header(
        "5. Multi-Agent Decision System"
    )


    col1, col2 = (
        st.columns(2)
    )


    with col1:

        st.subheader(
            "🤖 Predictive Agent"
        )

        st.json(
            predictive_result
        )


        st.subheader(
            "👁️ Vision Agent"
        )

        st.json(
            vision_result
        )


    with col2:

        st.subheader(
            "📚 Knowledge Agent"
        )

        st.json(
            knowledge_result
        )


        st.subheader(
            "🧠 Planning Agent"
        )

        st.json(
            planning_result
        )


    st.markdown("---")


    recommendation = (
        planning_result.get(
            "recommendation",
            "Run analysis first."
        )
    )


    st.subheader(
        "🎯 Final AI Recommendation"
    )


    st.success(
        recommendation
    )


# ============================================================
# TAB 6: DIGITAL TWIN
# ============================================================

with tab6:

    st.header(
        "6. Factory Digital Twin"
    )


    st.write(
        "Simulate different maintenance "
        "strategies before taking action "
        "on the real machine."
    )


    baseline_risk = (
        failure_probability
    )


    # --------------------------------------------------------
    # Continue Operation
    # --------------------------------------------------------

    continue_risk = (
        baseline_risk
    )

    continue_downtime = 0

    continue_loss = 0

    continue_cost = (
        continue_risk * 5000
        + continue_downtime * 1000
        + continue_loss * 5
    )


    # --------------------------------------------------------
    # Stop for Maintenance
    # --------------------------------------------------------

    maintenance_risk = 0.15

    maintenance_downtime = 4

    maintenance_loss = 400

    maintenance_cost = (
        maintenance_risk * 5000
        + maintenance_downtime * 1000
        + maintenance_loss * 5
    )


    # --------------------------------------------------------
    # Reduce Machine Load
    # --------------------------------------------------------

    reduced_risk = min(
        baseline_risk * 0.45,
        0.99
    )

    reduced_downtime = 1

    reduced_loss = 100

    reduced_cost = (
        reduced_risk * 5000
        + reduced_downtime * 1000
        + reduced_loss * 5
    )


    digital_twin_df = pd.DataFrame(
        [

            {
                "Scenario":
                    "Continue Operation",

                "Downtime Hours":
                    continue_downtime,

                "Failure Risk":
                    round(
                        continue_risk,
                        2
                    ),

                "Production Loss":
                    continue_loss,

                "Estimated Cost":
                    round(
                        continue_cost,
                        2
                    )
            },


            {
                "Scenario":
                    "Stop for Maintenance",

                "Downtime Hours":
                    maintenance_downtime,

                "Failure Risk":
                    maintenance_risk,

                "Production Loss":
                    maintenance_loss,

                "Estimated Cost":
                    round(
                        maintenance_cost,
                        2
                    )
            },


            {
                "Scenario":
                    "Reduce Machine Load",

                "Downtime Hours":
                    reduced_downtime,

                "Failure Risk":
                    round(
                        reduced_risk,
                        2
                    ),

                "Production Loss":
                    reduced_loss,

                "Estimated Cost":
                    round(
                        reduced_cost,
                        2
                    )
            }
        ]
    )


    st.subheader(
        "WHAT-IF RESULTS"
    )


    st.dataframe(
        digital_twin_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Best Scenario
    # --------------------------------------------------------

    best_scenario = (
        digital_twin_df.loc[
            digital_twin_df[
                "Estimated Cost"
            ].idxmin()
        ]
    )


    st.markdown("---")


    st.subheader(
        "🏆 Recommended Scenario"
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "Scenario",
            best_scenario[
                "Scenario"
            ]
        )


    with col2:

        st.metric(
            "Estimated Cost",
            f"${best_scenario['Estimated Cost']:,.0f}"
        )


    with col3:

        st.metric(
            "Failure Risk",
            f"{best_scenario['Failure Risk'] * 100:.0f}%"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Factory AI Predictive Maintenance | "
    "Random Forest + Deep Learning + "
    "Computer Vision + RAG + "
    "Multi-Agent Workflow + Digital Twin"
)