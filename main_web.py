import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Particle-Based Causal AI Decision Intelligence Dashboard",
    page_icon="🤖",
    layout="wide"
)


# =========================
# Custom CSS styling
# =========================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
}

.metric-card {
    background-color: #111;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #333;
}

.prediction-text {
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Load dataset
# =========================

df = pd.read_csv("particle_dataset_noisy.csv")


# =========================
# Model definition
# =========================

class UnifiedMultimodalModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.electrical_pathway = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.physical_pathway = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.dynamic_pathway = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3)
        )

    def forward(self, charge, mass, dynamic):

        e = self.electrical_pathway(charge)
        p = self.physical_pathway(mass)
        d = self.dynamic_pathway(dynamic)

        combined = torch.cat([e, p, d], dim=1)

        return self.fusion_layer(combined)


# Load model
model = UnifiedMultimodalModel()

model.load_state_dict(
    torch.load("unified_multimodal_model.pth", map_location="cpu"))

model.train()

classes = ["Electron", "Proton", "Neutron"]


# =========================
# Monte Carlo prediction
# =========================

def monte_carlo_prediction(charge, mass, dynamic):

    predictions = []

    with torch.no_grad():

        for _ in range(30):

            out = model(charge, mass, dynamic)

            prob = torch.softmax(out, dim=1)

            predictions.append(prob.numpy())

    predictions = np.array(predictions)

    mean = predictions.mean(axis=0)[0]
    var = predictions.var(axis=0)[0]

    return mean, var


# =========================
# UI Layout
# =========================

st.markdown('<div class="main-title">Multimodal AI Live Prediction Dashboard</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

prediction_box = col1.empty()
confidence_box = col2.empty()
uncertainty_box = col3.empty()

st.divider()

input_box = st.empty()

progress_bar = st.progress(0)


# =========================
# Streaming loop
# =========================

for i in range(len(df)):

    row = df.iloc[i]

    charge = torch.FloatTensor([[row["charge"]]])
    mass = torch.FloatTensor([[row["mass"]]])
    dynamic = torch.FloatTensor([[row["energy"], row["momentum"]]])

    mean, var = monte_carlo_prediction(charge, mass, dynamic)

    pred_class = classes[np.argmax(mean)]

    confidence = float(np.max(mean))
    uncertainty = float(np.mean(var))

    # Prediction display
    prediction_box.metric(
        label="Prediction",
        value=pred_class
    )

    # Confidence display
    confidence_box.metric(
        label="Confidence",
        value=f"{confidence:.3f}"
    )

    # Uncertainty display
    uncertainty_box.metric(
        label="Uncertainty",
        value=f"{uncertainty:.6f}"
    )

    # Progress bar
    progress_bar.progress(confidence)

    # Input data display
    input_box.dataframe(
        pd.DataFrame({
            "Feature": ["Charge", "Mass", "Energy", "Momentum"],
            "Value": [
                row["charge"],
                row["mass"],
                row["energy"],
                row["momentum"]
            ]
        }),
        use_container_width=True
    )

    time.sleep(0.10)
