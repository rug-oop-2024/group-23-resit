import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.regression.lasso_wrapper import LassoWrapper
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors)
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegressionModel)
from autoop.core.ml.model.classification.naive_bayes import NaiveBayesModel
from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.pipeline import Pipeline


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """Text writing helper that adds style"""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning "
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
dataset_names = [dataset.name for dataset in datasets]

# Step 1: Select Dataset
st.header("1. Select Dataset")
dataset_name = st.selectbox("Choose a dataset", dataset_names)
selected_artifact = next((artifact for artifact in datasets
                          if artifact.name == dataset_name), None)

# Check if the artifact's type is a dataset
if selected_artifact and selected_artifact.type == "dataset":
    selected_dataset = Dataset(
        name=selected_artifact.name,
        asset_path=selected_artifact.asset_path,
        data=selected_artifact.data,
    )
else:
    selected_dataset = None

if selected_dataset:
    df = selected_dataset.read()
    features = detect_feature_types(selected_dataset)
    feature_names = [feature.name for feature in features]
    feature_types = {feature.name: feature.type for feature in features}

    # Display dataset preview
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Step 2: Define Features and Target
    st.header("2. Define Features and Target")
    input_features = st.multiselect("Select input features",
                                    options=feature_names)
    target_feature = st.selectbox("Select target feature",
                                  options=feature_names)

    if target_feature:
        # Detect task type based on the target feature's type
        if feature_types[target_feature] == "categorical":
            task_type = "classification"
        else:
            task_type = "regression"
        st.write(f"Detected task type: {task_type}")

        input_feature_objs = [Feature(name=feature,
                                      type=feature_types[feature])
                              for feature in input_features]
        target_feature_obj = Feature(name=target_feature,
                                     type=feature_types[target_feature])

        # Step 3: Select Model (based on task type)
        st.header("3. Select Model")
        if task_type == "classification":
            model_options = {
                "knn": KNearestNeighbors,
                "logistic_regression": LogisticRegressionModel,
                "naive_bayes": NaiveBayesModel}
        else:
            model_options = {
                "multiple_linear_regression": MultipleLinearRegression,
                "lasso_wrapper": LassoWrapper,
                "ridge_regression": RidgeRegression}

        model_name = st.selectbox("Choose a model", list(model_options.keys()))
        model = model_options[model_name]()

        # Step 4: Select Dataset Split
        st.header("4. Select Dataset Split")
        split = st.slider("Select training/testing split", min_value=0.1,
                          max_value=0.9, value=0.8)

        # Step 5: Select Metrics
        st.header("5. Select Metrics")
        compatible_metrics = [
            metric
            for metric in METRICS
            if (
                task_type == "classification" and metric in [
                    "accuracy", "precision", "f1_score"
                ]
            ) or (
                task_type == "regression" and metric in [
                    "mean_squared_error", "mean_absolute_error", "r_squared"
                ]
            )
        ]
        selected_metrics = st.multiselect("Select metrics for evaluation",
                                          options=compatible_metrics)
        metrics_objs = [get_metric(metric) for metric in selected_metrics]

        # Step 6: Pipeline Summary
        st.header("6. Pipeline Summary")
        st.write("### Summary of Your Configurations")
        st.write(f"- **Dataset**: {selected_dataset.name}")
        st.write(f"- **Input Features**: {', '.join(input_features)}")
        st.write(f"- **Target Feature**: {target_feature}")
        st.write(f"- **Task Type**: {task_type}")
        st.write(f"- **Model**: {model_name}")
        st.write(f"- **Split**: {split * 100:.0f}% training /"
                 f"{100 - split * 100:.0f}% testing")
        st.write(f"- **Metrics**: {', '.join(selected_metrics)}")

        # Step 7: Train Model and Report Results
        st.header("7. Train the Model")
        if st.button("Train"):
            # Initialize and execute the pipeline
            pipeline = Pipeline(
                metrics=metrics_objs,
                dataset=selected_dataset,
                model=model,
                input_features=input_feature_objs,
                target_feature=target_feature_obj,
                split=split
            )
            results = pipeline.execute()

            # Display results
            st.write("### Results")
            for metric, result in results["metrics"]:
                st.write(f"{metric}: {result}")

            st.write("### Predictions")
            st.write("Train Set Predictions:", results["predictions"]["train"])
            st.write("Test Set Predictions:", results["predictions"]["test"])

            # Step 8: Save pipeline
            st.write("## Save the trained pipeline")

            pipeline_name = st.text_input("Enter a name for the pipeline")
            pipeline_version = st.text_input("Enter a version for the "
                                             "pipeline", "1.0.0")

            if st.button("Save Pipeline"):
                if pipeline_name and pipeline_version:
                    # Convert pipeline to artifact and save
                    artifact_registry = AutoMLSystem.get_instance().registry
                    pipeline_artifact = pipeline.to_artifact(
                        name=pipeline_name,
                        version=pipeline_version)
                    artifact_registry.register(pipeline_artifact)
                    st.success(f"Pipeline '{pipeline_name}' "
                               f"(version {pipeline_version}) saved "
                               "successfully!")
                else:
                    st.warning("Please provide a name and version for the "
                               "pipeline.")
