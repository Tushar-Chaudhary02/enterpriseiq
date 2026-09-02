"""Initial EnterpriseIQ application entry point."""

import json

from enterpriseiq.config import get_settings


def project_status() -> dict[str, str]:
    """Return basic information about the running project."""

    settings = get_settings()

    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_environment,
        "status": "ready",
    }


def main() -> None:
    """Display the current project status."""

    print(json.dumps(project_status(), indent=2))


if __name__ == "__main__":
    main()
