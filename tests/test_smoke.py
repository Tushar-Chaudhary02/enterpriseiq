"""Initial smoke tests for EnterpriseIQ."""

from enterpriseiq.main import project_status


def test_project_status_is_ready() -> None:
    """The project should report a valid ready state."""

    status = project_status()

    assert status["name"] == "EnterpriseIQ"
    assert status["version"] == "0.1.0"
    assert status["environment"] == "development"
    assert status["status"] == "ready"
