"""Initial smoke tests for EnterpriseIQ."""

from enterpriseiq import __version__
from enterpriseiq.main import project_status


def test_project_status_is_ready() -> None:
    """The project should report a valid ready state."""

    status = project_status()

    assert status["name"] == "EnterpriseIQ"
    assert __version__ == "0.8.0"
    assert status["version"] == __version__
    assert status["environment"] == "development"
    assert status["status"] == "ready"
