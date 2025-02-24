"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro.pipeline import Pipeline
# from century_health_assignment.pipelines import data_analysis as da
from century_health_assignment.pipelines.data_processing.pipeline import create_pipeline as de_pipeline

# from century-health-assignment.pipelines.data_engineering.pipeline import create_pipeline as de_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_engineering": de_pipeline(),
        # "data_analysis": da.create_pipeline()
        }


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
# #     # pipelines = find_pipelines()
# #     # pipelines["__default__"] = sum(pipelines.values())
# #     # return pipelines
    
# # def register_pipelines() -> dict[str, Pipeline]:
# #     return {"data_engineering": de_pipeline()}

