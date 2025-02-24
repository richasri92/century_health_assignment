import pytest
import pandas as pd
from century_health_assignment.pipelines.data_processing.nodes import split_symptoms



@pytest.fixture
def sample_symptoms():
    """Create a sample input DataFrame for testing."""
    data = {
        "patient_id": [1, 2],
        "observation": [
            "Rash:34;Joint Pain:39;Fatigue:9;Fever:12",
            "Rash:19;Joint Pain:44;Fatigue:48;Fever:15"
        ],
    }
    return pd.DataFrame(data)

def test_split_symptoms(sample_symptoms):
    """Test the split_symptoms function."""
    result = split_symptoms(sample_symptoms)

    # Check that new symptom columns exist
    assert "Rash" in result.columns
    assert "Joint Pain" in result.columns
    assert "Fatigue" in result.columns
    assert "Fever" in result.columns

    # Check that values are correctly extracted
    assert result.loc[0, "Rash"] == 34.0
    assert result.loc[1, "Fever"] == 15.0

    # Ensure original columns (except 'observation') are preserved
    assert "observation" not in result.columns
    assert "patient_id" in result.columns
