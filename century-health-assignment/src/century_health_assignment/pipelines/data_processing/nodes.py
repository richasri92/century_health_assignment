import matplotlib
matplotlib.use("Agg")

import pandas as pd
import plotly.express as px
# from kedro_datasets import CSVDataSet


from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import os

import pandas as pd
from scipy.stats import zscore
import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
# from kedro.config import ConfigLoader
from kedro.io import DataCatalog
import networkx as nx
import matplotlib.pyplot as plt
from kedro.framework.context import KedroContext
import networkx as nx
import matplotlib.pyplot as plt
# from kedro.config import TemplatedConfigLoader
from kedro.config.omegaconf_config import OmegaConfigLoader  # ✅ Use this instead

from pathlib import Path


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x

def link_datasets(raw_patients, raw_encounters, raw_symptoms, raw_conditions, raw_medications):
    return raw_patients, raw_encounters,raw_symptoms, raw_conditions, raw_medications

def create_master_dataset(patients, encounters,symptoms, conditions, medications):
    # patients = patients.rename(columns={"PATIENT_ID": "PATIENT"})
    
    
    df_joined = (
        patients
            .merge(encounters, on="patient_id", how="left")
            .merge(symptoms, on="patient_id", how="left")
            .merge(conditions, on="patient_id", how="left")
            .merge(medications, on="patient_id", how="left")
    )
    return df_joined

def add_patient_gender(patients, patient_gender):
    if 'GENDER_x' in patients.columns:  # Check if 'GENDER' exists before dropping
        patients = patients.drop(columns=['GENDER_x'])

    return patients.merge(patient_gender, left_on="PATIENT_ID", right_on="Id", how="left")

COLUMN_MAPPINGS = {
    "patients": {
        "PATIENT_ID": "patient_id",       # Unique identifier for the patient
        "NAME": "full_name",              # Patient's full name
        "DOB": "birth_date",              # Date of birth
        "GENDER": "sex",                  # Gender of the patient
        # Add other mappings as necessary
    },
    "encounters": {
        "ENCOUNTER_ID": "Id",  
        "PATIENT": "patient_id",      
        "ENCOUNTERCLASS": "encounter_type"
    },
    "symptoms": {  # Corresponds to "Observations" in Tuva
        "OBSERVATION_ID": "observation_id",  # Concatenated the patient id and pathology to create a unique key
        "PATIENT": "patient_id",          
        "SYMPTOMS": "observation"
    },
    "conditions": {
        "CODE": "condition_id",    
        "PATIENT": "patient_id",       
        "ENCOUNTER": "encounter_id"
    },
    "medications": {
        "CODE": "medication_id", 
        "PATIENT": "patient_id",
        "ENCOUNTER": "encounter_id"
        
    },
    
}
def standardize_as_per_TUVA(patients, encounters, symptoms, conditions, medications):
    
    symptoms['OBSERVATION_ID'] = symptoms['PATIENT']+"--"+symptoms['PATHOLOGY']
    patients['NAME'] =  patients['PREFIX']+ " "+patients['FIRST']+" "+patients['LAST']
    
    def standardize_column_names(df, dataset_name):
        """Rename DataFrame columns according to Tuva Data Model."""
        if dataset_name in COLUMN_MAPPINGS:
            return df.rename(columns=COLUMN_MAPPINGS[dataset_name])
        return df  # If no mapping found, return original DataFrame

    # Example Usage
    patients = standardize_column_names(patients, "patients")
    encounters = standardize_column_names(encounters, "encounters")
    symptoms = standardize_column_names(symptoms, "symptoms")  # Maps to "Observations"
    conditions = standardize_column_names(conditions, "conditions")
    medications = standardize_column_names(medications, "medications")
    
    return patients,encounters,symptoms,conditions,medications
    

def split_symptoms(symptoms):
    symptom_dicts = symptoms["observation"].apply(lambda x: dict(item.split(":") for item in x.split(";")))
    symptom_df = pd.DataFrame(symptom_dicts.tolist()).astype(float)

    # Merge with original DataFrame
    df = pd.concat([symptoms.drop(columns=["observation"]), symptom_df], axis=1)
    return df




def analyze_patient_data(patients_df: pd.DataFrame, medications_df: pd.DataFrame):
    """
    Perform data analysis on patient and medication datasets.

    Args:
        patients_df (pd.DataFrame): DataFrame containing patient data.
        medications_df (pd.DataFrame): DataFrame containing medication data.

    Returns:
        dict: Dictionary of analysis results and visualizations.
    """

    # 1. How many distinct patients are in the dataset?
    num_patients = patients_df["PATIENT_ID"].nunique()
    print(patients_df.columns)

    # 2. Plot distinct medications over time
    medications_df["START"] = pd.to_datetime(medications_df["START"])  # Ensure proper datetime format
    med_count_over_time = medications_df.groupby("START")["CODE"].nunique().reset_index()
    medication_plot = px.line(
        med_count_over_time,
        x="START",
        y="CODE",
        title="Distinct Medications Over Time",
        labels={"MEDICATION": "Number of Distinct Medications"}
    )
    medication_plot.write_html("data/08_reporting/medication_plot.html")  # Save as an interactive HTML file

    # 3. Pie chart for patient racial categories and gender distribution
    race_gender_count = patients_df.groupby(["RACE", "GENDER_y"]).size().reset_index(name="count")
    race_gender_pie = px.pie(
        race_gender_count,
        names="RACE",
        values="count",
        title="Patient Distribution by Race",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    race_gender_pie.write_html("data/08_reporting/race_gender_pie_plot.html")  # Save as an interactive HTML file

    # 4. Percentage of patients with all 4 symptom categories ≥ 30
    # symptoms_cols = ["Rash", "Joint Pain", "Fatigue", "Fever"]
    # patients_df[symptoms_cols] = patients_df["SYMPTOMS"].str.split(";", expand=True).applymap(lambda x: int(x.split(":")[-1]))

    # high_severity_patients = patients_df[
    #     (patients_df["Rash"] >= 30) &
    #     (patients_df["Joint Pain"] >= 30) &
    #     (patients_df["Fatigue"] >= 30) &
    #     (patients_df["Fever"] >= 30)
    # ]

    # high_symptom_percentage = (len(high_severity_patients) / len(patients_df)) * 100

    return {
        "num_patients": num_patients,
        "medication_plot": medication_plot,
        "race_gender_pie": race_gender_pie,
        # "high_symptom_percentage": high_symptom_percentage
    }


def join_customers_orders(patients: pd.DataFrame, encounters: pd.DataFrame) -> pd.DataFrame:
    """Create a relationship between customers and orders."""
    return patients.merge(encounters, on="PATIENT_ID", how="left")



def extract_dataset_relationships():
    """Extract metadata from Kedro's catalog.yml."""
    config_loader = OmegaConfigLoader(conf_source="conf/base")
    catalog_config = config_loader.get("catalog")

    datasets = {}
    for dataset_name, dataset_info in catalog_config.items():
        metadata = dataset_info.get("metadata", {})
        datasets[dataset_name] = {
            "primary_key": metadata.get("primary_key"),
            "foreign_keys": metadata.get("foreign_keys", []),
        }
    return datasets



def generate_er_diagram(datasets):
    """Generate an ER diagram and return a PIL Image object for Kedro."""
    G = nx.DiGraph()

    # Add nodes (tables) and edges (relationships)
    for table, info in datasets.items():
        G.add_node(table, label=f"{table}\nPK: {info['primary_key']}", shape="box")

    for table, info in datasets.items():
        for fk in info["foreign_keys"]:
            for parent_table, parent_info in datasets.items():
                if fk == parent_info["primary_key"]:
                    G.add_edge(parent_table, table, label=f"{parent_table}.{fk} → {table}.{fk}")

    # Draw ER Diagram
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, edge_color="black", font_size=10)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Save ER diagram to a temporary file
    output_path = "/Users/richa_srivastava/new_KOHLS/price-optimizer/century-health-assignment/data/02_intermediate/er_diagram.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()  # Prevent Matplotlib memory issues

    # Load the saved image as a PIL Image object
    image = Image.open(output_path)
    return image  # Return the Image object, not the file path


def run_er_diagram_pipeline():
    """Main function to run ER diagram pipeline."""
    datasets = extract_dataset_relationships()
    generate_er_diagram(datasets)


def visualize_relationships(
    patients: pd.DataFrame, 
    # symptoms: pd.DataFrame, 
    # medications: pd.DataFrame,
    # conditions: pd.DataFrame,
    encounters: pd.DataFrame
    ) -> None:
    """
    Creates a relationship graph between datasets.
    """
    G = nx.DiGraph()

    # Adding nodes
    G.add_node("patients", color="red", shape="o")
    G.add_node("symptoms", color="blue", shape="s")
    G.add_node("medications", color="green", shape="d")
    G.add_node("merged_data", color="purple", shape="*")

    # Adding edges (relationships)
    G.add_edge("patients", "merged_data")
    G.add_edge("symptoms", "merged_data")
    G.add_edge("medications", "merged_data")

    # Draw the graph
    plt.figure(figsize=(6, 4))
    colors = [G.nodes[n]["color"] for n in G.nodes()]
    shapes = [G.nodes[n]["shape"] for n in G.nodes()]

    nx.draw(G, with_labels=True, node_color=colors, edge_color="gray", node_size=3000, font_size=10, font_color="white")
    plt.title("DataFrame Relationships")
    plt.show()



def detect_outliers(df: pd.DataFrame, col: str, threshold: float = 3):
    """Detect outliers in a single column based on Z-score."""
    
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return None  # Return None if column doesn't exist or isn't numeric

    df_no_na = df[col].dropna()  # Remove NaNs for proper Z-score calculation
    z_scores = zscore(df_no_na)
    
    outlier_values = df_no_na[(z_scores > threshold) | (z_scores < -threshold)].tolist()
    
    return outlier_values

def check_duplicate_rows(df: pd.DataFrame):
    """Checks for duplicate rows in the dataset."""
    return df[df.duplicated()]


def check_unexpected_categories(df: pd.DataFrame, column, expected_categories):
    """Checks for unexpected values in a categorical column."""
    unique_values = set(df[column].dropna().unique())
    unexpected = unique_values - set(expected_categories)
    return list(unexpected)


def check_value_ranges(df: pd.DataFrame, column_ranges=None):
    """
    Checks whether values in all numeric columns fall within specified valid ranges.
    
    Args:
        df (pd.DataFrame): The DataFrame to check.
        column_ranges (dict): Dictionary specifying valid ranges, e.g., 
                              {"AGE_BEGIN": (0, 120), "NUM_SYMPTOMS": (0, 50)}
    
    Returns:
        dict: Columns with values outside the valid range.
    """

    if column_ranges is None:
        column_ranges = {
            "AGE_BEGIN": (0, 120),
            "NUM_SYMPTOMS": (0, 50)  # Example: Modify as needed
        }

    out_of_range = {}

    for col in df.select_dtypes(include=["number"]).columns:  # Apply to numeric columns
        if col in column_ranges:
            min_val, max_val = column_ranges[col]
            out_of_range[col] = df[(df[col] < min_val) | (df[col] > max_val)][col].tolist()
        else:
            out_of_range[col] = None  # No range defined for this column

    return out_of_range

def analyze_missing_patterns(df: pd.DataFrame):
    """Finds patterns in missing values."""
    return df.isnull().groupby(df["RACE"]).sum()


def check_value_distribution(df: pd.DataFrame, column):
    """Checks if a numeric column has an unusual distribution."""
    return df[column].describe()


def check_symptom_vs_age(df: pd.DataFrame):
    """Checks if older patients tend to have more symptoms."""
    return df[df["AGE_BEGIN"] > 70].groupby("NUM_SYMPTOMS").count()


def validate_dataframe(df: pd.DataFrame):
    """Perform advanced QC checks and return results in a structured DataFrame."""
    
    qc_results = {
        "Column": df.columns.tolist(),  # Column names for reference
        "Missing_Values": df.isnull().sum().tolist(),
        "Fully_Null_Columns": [df[col].isnull().all() for col in df.columns],
        "Dtype": [df[col].dtype.name for col in df.columns],
        "Negative_Values": [(df[col] < 0).sum() if col in df.select_dtypes(include=['number']).columns else None for col in df.columns],
        "Duplicate_Rows": [len(check_duplicate_rows(df)) if i == 0 else None for i in range(len(df.columns))],
        "Outliers": [detect_outliers(df, col) if col in ("AGE_BEGIN", "NUM_SYMPTOMS") else None for col in df.columns],
        "Unexpected_Categories": [check_unexpected_categories(df, col, ["Asian", "Black", "White", "Hispanic"]) if col == "RACE" else None for col in df.columns],
        "Out of Range Values": [check_value_ranges(df) if i == 0 else None for i in range(len(df.columns))],
        # "Missing_Patterns": [analyze_missing_patterns(df) if i == 0 else None for i in range(len(df.columns))]
    }

    # Convert the dictionary into a structured DataFrame
    qc_df = pd.DataFrame(qc_results)

    return qc_df


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def connect_data(patients: pd.DataFrame, symptoms: pd.DataFrame, medications: pd.DataFrame) -> pd.DataFrame:
    """Combines patient, symptoms, and medication datasets based on a common key (e.g., patient_id)."""
    df = patients.merge(symptoms, on="patient_id", how="left")
    df = df.merge(medications, on="patient_id", how="left")
    return df


def preprocess_dataframes(medications, symptoms, patients, conditions, encounters):
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    # medications["iata_approved"] = _is_true(companies["iata_approved"])
    # medications["company_rating"] = _parse_percentage(companies["company_rating"])
    # medications['formula_app'] = medications['BASE_COST'] * medications['DISPENSES']
    # medications['correct'] = medications['formula_app'].astype(int) == medications['TOTALCOST'].astype(int)
    return medications,symptoms,patients,conditions,encounters


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
