"""Tests for raw dataset validation."""

import csv
from pathlib import Path

import pytest

from enterpriseiq.data.download import EXPECTED_COLUMNS, validate_dataset


def write_test_dataset(
    path: Path,
    columns: list[str],
    row_count: int,
) -> None:
    """Create a small CSV file for isolated unit testing."""

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        for row_number in range(row_count):
            row = {column: "" for column in columns}

            if "customerID" in row:
                row["customerID"] = f"TEST-{row_number}"

            writer.writerow(row)


def test_validate_dataset_accepts_valid_schema(tmp_path: Path) -> None:
    """A dataset with the expected schema and row count should pass."""

    dataset_path = tmp_path / "valid.csv"
    columns = list(EXPECTED_COLUMNS)

    write_test_dataset(dataset_path, columns, row_count=2)

    validated_rows = validate_dataset(
        dataset_path,
        expected_columns=columns,
        expected_rows=2,
    )

    assert validated_rows == 2


def test_validate_dataset_rejects_missing_column(tmp_path: Path) -> None:
    """Validation should fail when a required column is missing."""

    dataset_path = tmp_path / "missing-column.csv"
    columns = list(EXPECTED_COLUMNS[:-1])

    write_test_dataset(dataset_path, columns, row_count=2)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(
            dataset_path,
            expected_columns=EXPECTED_COLUMNS,
            expected_rows=2,
        )


def test_validate_dataset_rejects_wrong_row_count(tmp_path: Path) -> None:
    """Validation should fail when the record count is incorrect."""

    dataset_path = tmp_path / "wrong-row-count.csv"
    columns = list(EXPECTED_COLUMNS)

    write_test_dataset(dataset_path, columns, row_count=1)

    with pytest.raises(ValueError, match="Expected 2 rows"):
        validate_dataset(
            dataset_path,
            expected_columns=columns,
            expected_rows=2,
        )
