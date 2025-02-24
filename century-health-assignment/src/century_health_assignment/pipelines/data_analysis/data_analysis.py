import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from kedro.extras.datasets.pandas import CSVDataSet

# Load dataset (Assuming it's a CSV file)
dataset_path = "/Users/richa_srivastava/new_KOHLS/price-optimizer/century-health-assignment/data/01_raw/symptoms.csv"
gender_patients_path = "/Users/richa_srivastava/new_KOHLS/price-optimizer/century-health-assignment/data/01_raw/patient_gender.csv"
medications= pd.read_csv("/Users/richa_srivastava/new_KOHLS/price-optimizer/century-health-assignment/data/01_raw/medications.csv")

gender_patients = pd.read_csv(gender_patients_path)
data = pd.read_csv(dataset_path)

data = (
    data
    .merge(gender_patients, left_on="PATIENT", right_on="Id", how="left")
    .merge(medications, on = 'PATIENT', how="left")
)



### 1️⃣ Count Distinct Patients
num_patients = data["PATIENT"].nunique()
print(f"Total distinct patients: {num_patients}")

### 2️⃣ Plot Distinct Medications Over Time
plt.figure(figsize=(12, 6))
if "CODE" in data.columns and "START" in data.columns:
    data["DATE"] = pd.to_datetime(data["START"])  # Convert to datetime
    med_counts = data.groupby("START")["CODE"].nunique()

    sns.lineplot(x=med_counts.index, y=med_counts.values, marker="o")
    plt.xlabel("Date")
    plt.ylabel("Number of Distinct Medications")
    plt.title("Distinct Medications Over Time")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("MEDICATIONS or DATE column missing. Skipping plot.")

### 3️⃣ Pie Chart for Racial & Gender Distribution
plt.figure(figsize=(10, 5))

if "RACE" in data.columns and "GENDER" in data.columns:
    race_counts = data["RACE"].value_counts()
    gender_counts = data["GENDER"].value_counts()

    # Pie Chart for Race
    plt.subplot(1, 2, 1)
    plt.pie(race_counts, labels=race_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Patient Distribution by Race")

    # Pie Chart for Gender
    plt.subplot(1, 2, 2)
    plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Patient Distribution by Gender")

    plt.show()
else:
    print("RACE or GENDER column missing. Skipping pie chart.")

### 4️⃣ Percentage of Patients with All Symptoms ≥ 30
symptom_cols = ["Rash", "Joint Pain", "Fatigue", "Fever"]

if all(col in data.columns for col in symptom_cols):
    severe_cases = data[(data[symptom_cols] >= 30).all(axis=1)]
    percentage = (len(severe_cases) / len(data)) * 100
    print(f"Percentage of patients with all symptoms ≥ 30: {percentage:.2f}%")
else:
    print("Some symptom columns are missing.")

### 5️⃣ Save Output to Kedro Catalog
# print(percentage)
# print(severe_cases)

# output_dataset = CSVDataSet(filepath="data/output_analysis.csv", save_args={"index": False})
output_data = pd.DataFrame({"Metric": ["Distinct Patients", "Severe Cases (%)"], "Value": [num_patients, percentage]})
output_data.save("/Users/richa_srivastava/new_KOHLS/price-optimizer/century-health-assignment/data/08_reporting/output_analysis.csv")
