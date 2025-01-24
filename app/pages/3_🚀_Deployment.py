import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
import pickle
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric, METRICS

st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment - Load and Predict with Saved Pipelines")

# Access the artifact registry
artifact_registry = AutoMLSystem.get_instance().registry

# Step 1: Select a saved pipeline
st.header("1. Select a Saved Pipeline")
saved_pipelines = artifact_registry.list(type="pipeline")
if not saved_pipelines:
    st.write("No pipelines found. Please train a pipeline first.")
    st.stop()

pipeline_names = [pipeline.name for pipeline in saved_pipelines]
selected_pipeline_name = st.selectbox("Choose a pipeline", pipeline_names)

# Step 2: Load and Show Pipeline Summary
if selected_pipeline_name:
    selected_pipeline_artifact = next(
        pipeline for pipeline in saved_pipelines
        if pipeline.name == selected_pipeline_name)
    selected_artifact = selected_pipeline_artifact.id
    loaded_pipeline = artifact_registry.get(selected_artifact)
    pipeline_data = pickle.loads(loaded_pipeline.data)
    pipeline = Pipeline(
        metrics=pipeline_data["metrics"],
        dataset=pipeline_data["dataset"],
        model=pipeline_data["model"],
        input_features=pipeline_data["input_features"],
        target_feature=pipeline_data["target_feature"],
        split=pipeline_data["split"],
    )
    st.session_state["pipeline"] = pipeline
    st.write("Pipeline loaded successfully.")

# Step 3: Upload CSV for Predictions
st.header("2. Upload a CSV for Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file and selected_pipeline_name:
    df = pd.read_csv(uploaded_file)
    dataset_name = uploaded_file.name.split(".")[0]
    dataset = Dataset.from_dataframe(
            data=df, name=dataset_name, asset_path=uploaded_file.name
        )
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("Run Predictions"):
        # Perform predictions
        pipeline._dataset = dataset
        # Check if Dataset is compatible with pipeline
        ok = 0
        data = dataset.read()
        for name in data.columns:
            for feature in pipeline._input_features:
                if name == feature.name:
                    ok = 1
        if ok == 1:
            predictions = pipeline.execute()
            st.write("### Predictions")
            st.write(predictions)
        else:
            st.write("dataset does not contain required features")

        # Allow the user to compare predictions with ground truth
        st.write("## Compare Predictions with Ground Truth")
        ground_truth_input = st.text_area(
            "Enter ground truth values (comma-separated)",
            placeholder="E.g., 120000, 140000, 130000..."
        )
        if ground_truth_input:
            try:
                ground_truth = list(map(float,
                                        ground_truth_input.split(",")))
                predictions = predictions["predictions"]["test"]
                task_type = pipeline._model.type
                # Allow metric selection for comparison
                st.write("### Select Metrics for Comparison")
                compatible_metrics = [
                    metric
                    for metric in METRICS
                    if (
                        task_type == "classification" and metric in [
                            "accuracy", "precision", "f1_score"
                        ]
                    ) or (
                        task_type == "regression" and metric in [
                            "mean_squared_error", "mean_absolute_error",
                            "r_squared"
                        ]
                    )
                ]

                selected_comparison_metrics = st.multiselect(
                    "Select metrics for comparison",
                    options=compatible_metrics,
                    default=compatible_metrics[:1]
                )
                comparison_metrics = [get_metric(metric) for metric
                                      in selected_comparison_metrics]

                # Compare predictions
                comparison_results = pipeline.compare_predictions(
                    y_true=ground_truth,
                    y_pred=predictions,
                    metrics=comparison_metrics
                )

                st.write("### Comparison Results")
                for metric_name, value in comparison_results.items():
                    st.write(f"{metric_name}: {value:.4f}")
            except ValueError:
                st.error("Please ensure the ground truth values are"
                         "numeric and properly formatted.")
