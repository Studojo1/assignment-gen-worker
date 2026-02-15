import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.graph_data_validator import verify_data_point, validate_graph_data
from src.tools.common.graph import GraphSpecification

def test_validation():
    print("Testing graph validation logic...")
    
    # Test Case 1: Conceptual Graph (Risk-Return)
    content = """
    The relationship between risk and return is fundamental in finance.
    Generally, low risk investments offer low return, while high risk investments offer high return.
    Medium risk investments typically provide medium return.
    """
    
    data_points = {
        "Low Risk": "Low Return",
        "Medium Risk": "Medium Return",
        "High Risk": "High Return"
    }
    
    print("\nTest Case 1: Conceptual Graph (Risk-Return)")
    for label, value in data_points.items():
        result = verify_data_point(label, value, content)
        print(f"  {label} = {value}: Verified={result.is_verified}, Score={result.confidence_score}, Source={result.source}")
        if not result.is_verified:
            print(f"    FAILED: {result.details}")

    # Test Case 2: Numeric Graph (Standard)
    content_numeric = """
    The company's revenue grew significantly.
    2020: $100 million
    2021: $150 million
    2022: $200 million
    """
    
    data_points_numeric = {
        "2020": 100,
        "2021": 150,
        "2022": 200
    }
    
    print("\nTest Case 2: Numeric Graph")
    for label, value in data_points_numeric.items():
        result = verify_data_point(label, value, content_numeric)
        print(f"  {label} = {value}: Verified={result.is_verified}, Score={result.confidence_score}, Source={result.source}")
        if not result.is_verified:
            print(f"    FAILED: {result.details}")

    # Test Case 3: Full Graph Specification Validation
    spec = GraphSpecification(
        title="Risk-Return Trade-off",
        graph_type="scatter",
        data_points=data_points,
        rationale="Illustrates the concept",
        x_axis_label="Risk",
        y_axis_label="Return"
    )
    
    is_valid, score, issues = validate_graph_data(spec, content)
    print(f"\nFull Graph Validation: Valid={is_valid}, Score={score}")
    if issues:
        print(f"  Issues: {issues}")

if __name__ == "__main__":
    test_validation()
