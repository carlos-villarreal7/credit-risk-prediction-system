from src.inference import segment_risk


def test_segment_risk_boundaries() -> None:
    assert segment_risk(0.05) == "Low"
    assert segment_risk(0.20) == "Medium"
    assert segment_risk(0.49) == "Medium"
    assert segment_risk(0.50) == "High"
