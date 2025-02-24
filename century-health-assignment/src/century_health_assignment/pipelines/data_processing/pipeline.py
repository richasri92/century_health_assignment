from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  (validate_dataframe, 
visualize_relationships, preprocess_dataframes,join_customers_orders,extract_dataset_relationships, generate_er_diagram,
create_master_dataset, add_patient_gender,standardize_as_per_TUVA,split_symptoms, link_datasets, analyze_patient_data
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # TASK: 1 **Link to the 5 datasets
            node(
                func = link_datasets,
                inputs = ["raw_patients", "raw_encounters", "raw_symptoms", "raw_conditions", "raw_medications"],
                outputs =["patients", "encounters", "symptoms", "conditions", "medications"],
                name = "link_datasets"
            ),
            
            # TASK: 2(ii). **Add patient gender as 6th dataset
            node(
                func = add_patient_gender,
                inputs = ['patients', "patients_gender"],
                outputs = "patients_with_gender",
                name = "clean_gender_info"
                ),
            #Task: 3(i) : TUVA Input layer re-structuing
            node(
                func = standardize_as_per_TUVA,
                inputs = ["patients_with_gender", "encounters", "symptoms", "conditions", "medications"],
                outputs = [
                    "patients_intermediate", 
                    "encounters_intermediate", 
                    "symptoms_intermediate",
                    "conditions_intermediate",
                     "medications_intermediate"
                     ],
                name = "standardize_as_per_TUVA"
            ),
            #Task: 3(ii) : Split symptoms column and break into indivisual columns
            node(
                func = split_symptoms,
                inputs = ['symptoms_intermediate'],
                outputs = 'symptoms_elaborate',
                name = "split_symptoms"
            ),
            # TASK: 4. Merge above datasets
            node(
                func = create_master_dataset,
                inputs=[
                    "patients_intermediate",
                    "encounters_intermediate",
                    "symptoms_elaborate", 
                    "conditions_intermediate", 
                    "medications_intermediate"],
                outputs = "master_df",
                name = "create_master_dataset"
            ),
            
            # # ANALYSE PATIENT DATA:
            # node(
            #     func = analyze_patient_data,
            #     inputs = ["patients_with_gender", "medications"],
            #     outputs="analysis_results",
            #     name="analyze_patient_data_node"
            # ),
            
            # node(
            #     func=validate_dataframe,
            #     inputs="symptoms",
            #     outputs="symptoms_QC",
            #     name="validate_dataframe_symptoms",
            # ),

            
        #     node(
        #     func=extract_dataset_relationships,
        #     inputs=None,
        #     outputs="dataset_relationships",
        #     name="extract_dataset_relationships_node",
        # ),
        # node(
        #     func=generate_er_diagram,
        #     inputs="dataset_relationships",
        #     outputs="er_diagram_file",
        #     name="generate_er_diagram_node",
        # )
        
        ]
    )
