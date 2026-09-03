"""Download and validate the raw customer-churn dataset."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Sequence
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

EXPECTED_ROWS = 7043

EXPECTED_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"


def validate_dataset(
    path: Path,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    expected_rows: int = EXPECTED_ROWS,
) -> int:
    """Validate the dataset's headers and number of records."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("Dataset does not contain a header row.")

        observed_columns = set(reader.fieldnames)
        required_columns = set(expected_columns)

        missing_columns = required_columns - observed_columns
        unexpected_columns = observed_columns - required_columns

        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

        if unexpected_columns:
            raise ValueError(f"Dataset contains unexpected columns: {sorted(unexpected_columns)}")

        row_count = sum(1 for _ in reader)

    if row_count != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, but found {row_count} rows.")

    return row_count


def download_dataset(
    destination: Path = DEFAULT_DATASET_PATH,
    *,
    force: bool = False,
) -> Path:
    """Download the dataset only when it is not already available."""

    if destination.exists() and not force:
        validate_dataset(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp")

    try:
        with (
            urlopen(DATASET_URL, timeout=60) as response,
            temporary_path.open("wb") as output_file,
        ):
            shutil.copyfileobj(response, output_file)

        validate_dataset(temporary_path)
        temporary_path.replace(destination)

    except (OSError, URLError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Dataset download failed: {error}") from error

    return destination


def main() -> None:
    """Download the raw dataset and report its location."""

    dataset_path = download_dataset()
    row_count = validate_dataset(dataset_path)

    print(f"Dataset ready: {dataset_path}")
    print(f"Validated rows: {row_count}")
    print(f"Validated columns: {len(EXPECTED_COLUMNS)}")


if __name__ == "__main__":
    main()
