import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()
artifact_registry = automl.registry

st.set_page_config(page_title="Dataset Management", layout="wide")
st.title("Manage Datasets")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    dataframe = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded dataset:", dataframe.head())

    dataset_name = st.text_input("Enter a name for the dataset artifact",
                                 value=uploaded_file.name.split('.')[0])

    asset_path = f"./{dataset_name}.csv"

    dataset = Dataset.from_dataframe(dataframe, name=dataset_name,
                                     asset_path=asset_path, version="1.0.0")

    if st.button("Save Dataset"):
        artifact_registry.register(dataset)
        st.success(f"Dataset '{dataset_name}' has been successfully uploaded"
                   f" and saved!")
