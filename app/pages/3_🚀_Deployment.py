import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment - Load and Predict with Saved Pipelines")

# Access the artifact registry
artifact_registry = AutoMLSystem.get_instance().registry

# Step 1: Select a saved pipeline
st.header("1. Select a Saved Pipeline")
saved_pipelines = artifact_registry.list(type="pipeline")
pipeline_names = [pipeline.name for pipeline in saved_pipelines]
selected_pipeline_name = st.selectbox("Choose a pipeline", pipeline_names)

# Step 2: Load and Show Pipeline Summary
if selected_pipeline_name:
    selected_pipeline_artifact = next(
        pipeline for pipeline in saved_pipelines
        if pipeline.name == selected_pipeline_name)
    loaded_pipeline = Pipeline.from_artifact(selected_pipeline_artifact)
    st.write("### Pipeline Summary")
    st.write(f"**Name**: {loaded_pipeline.name}")
    st.write(f"**Version**: {loaded_pipeline.version}")
    st.write(f"**Model Type**: {loaded_pipeline.model.type}")
    st.write(f"**Input Features**: "
             f"{[feature.name for feature in loaded_pipeline.input_features]}")
    st.write(f"**Target Feature**: {loaded_pipeline.target_feature.name}")
    metrics_list = ", ".join(
        metric.__class__.__name__ for metric in loaded_pipeline.metrics
    )
    st.write(f"**Metrics**: [{metrics_list}]")

# Step 3: Upload CSV for Predictions
st.header("2. Upload a CSV for Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file and selected_pipeline_name:
    prediction_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(prediction_data.head())

    if st.button("Run Predictions"):
        # Perform predictions
        predictions = loaded_pipeline.predict(prediction_data)
        prediction_data["Predictions"] = predictions
        st.write("### Predictions")
        st.dataframe(prediction_data)
        # Optionally, allow download of predictions as CSV
        csv_data = prediction_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions as CSV", data=csv_data,
                           file_name="predictions.csv", mime="text/csv")
