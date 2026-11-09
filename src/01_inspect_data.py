"""
src/01_inspect_data.py
======================
Step 1 – Data Inspection & Provenance for RTO Shield AI - Model 1.

Loads the raw DataCo Smart Supply Chain CSV, prints a detailed inspection
report, and writes a SHA-256 provenance record to outputs/data_provenance.txt.

Usage:
    python src/01_inspect_data.py

No model code. No data modification. Read-only.
"""

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV      = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PROVENANCE   = OUTPUTS_DIR / "data_provenance.txt"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns that are FORBIDDEN as model features (post-dispatch leakage or PII)
# ---------------------------------------------------------------------------
FORBIDDEN_FEATURES = [
    "Days for shipping (real)",   # actual outcome – pure leakage
    "Delivery Status",            # actual outcome – pure leakage
    "Order Status",               # actual outcome – pure leakage
    "shipping date (DateOrders)", # post-dispatch timestamp
    "Benefit per order",          # financial outcome, known after delivery
    "Order Profit Per Order",     # financial outcome
    "Order Item Profit Ratio",    # financial outcome
    # Customer PII
    "Customer Fname",
    "Customer Lname",
    "Customer Email",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Product Image",
    "Product Description",
]

# ---------------------------------------------------------------------------
# Columns that MUST exist in the dataset
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    # Target
    "Late_delivery_risk",
    # COD proxy
    "Type",
    # Date column for chronological split
    "order date (DateOrders)",
    # Candidate features (pre-dispatch, non-PII)
    "Days for shipment (scheduled)",
    "Sales per customer",
    "Category Id",
    "Category Name",
    "Customer City",
    "Customer Country",
    "Customer Id",
    "Customer Segment",
    "Customer State",
    "Department Id",
    "Department Name",
    "Latitude",
    "Longitude",
    "Market",
    "Order City",
    "Order Country",
    "Order Customer Id",
    "Order Id",
    "Order Item Cardprod Id",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Id",
    "Order Item Product Price",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Region",
    "Order State",
    "Order Zipcode",
    "Product Card Id",
    "Product Category Id",
    "Product Name",
    "Product Price",
    "Product Status",
    "Shipping Mode",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sha256_of_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load the raw CSV with correct encoding; strip whitespace from column names."""
    df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def print_section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def check_required_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (present, missing) lists for REQUIRED_COLUMNS."""
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return present, missing


def check_forbidden_columns(df: pd.DataFrame) -> list[str]:
    """Return which FORBIDDEN_FEATURES actually exist in the dataset."""
    return [c for c in FORBIDDEN_FEATURES if c in df.columns]


# ---------------------------------------------------------------------------
# Main inspection routine
# ---------------------------------------------------------------------------

def inspect(df: pd.DataFrame) -> None:

    # -- Shape ---------------------------------------------------------------
    print_section("1. Dataset Shape")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")

    # -- Stripped column names -----------------------------------------------
    print_section("2. All Column Names (after stripping whitespace)")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col}")

    # -- Data types ----------------------------------------------------------
    print_section("3. Data Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):<15}: {count} column(s)")
    print()
    for col, dtype in df.dtypes.items():
        print(f"  {col:<45}  {dtype}")

    # -- Unique Order Ids ----------------------------------------------------
    print_section("4. Unique Order Id Count")
    if "Order Id" in df.columns:
        n_orders = df["Order Id"].nunique()
        print(f"  Unique Order Id values : {n_orders:,}")
        print(f"  Total rows             : {df.shape[0]:,}")
        print(f"  Rows per order (avg)   : {df.shape[0] / n_orders:.2f}")
    else:
        print("  [WARN] 'Order Id' column NOT FOUND.")

    # -- Target column -------------------------------------------------------
    print_section("5. Target Column: Late_delivery_risk")
    if "Late_delivery_risk" in df.columns:
        vc = df["Late_delivery_risk"].value_counts(dropna=False)
        total = len(df)
        for val, cnt in vc.items():
            print(f"  {val}  →  {cnt:,} rows  ({cnt/total*100:.1f}%)")
    else:
        print("  [WARN] 'Late_delivery_risk' column NOT FOUND.")

    # -- COD proxy -----------------------------------------------------------
    print_section("6. COD Proxy: Type == 'CASH'")
    if "Type" in df.columns:
        vc = df["Type"].value_counts(dropna=False)
        for val, cnt in vc.items():
            marker = "  ← COD proxy" if val == "CASH" else ""
            print(f"  {str(val):<20}  {cnt:,} rows{marker}")
    else:
        print("  [WARN] 'Type' column NOT FOUND.")

    # -- Required columns check ----------------------------------------------
    print_section("7. Required Columns Check")
    present, missing = check_required_columns(df)
    print(f"  [OK] Present : {len(present)} / {len(REQUIRED_COLUMNS)}")
    if missing:
        print(f"  [MISSING] Missing : {len(missing)}")
        for c in missing:
            print(f"       - {c}")
    else:
        print("  [OK] All required columns are present.")

    # -- Forbidden columns check ---------------------------------------------
    print_section("8. Forbidden Feature Columns (must NOT be used as features)")
    found_forbidden = check_forbidden_columns(df)
    print(f"  Found {len(found_forbidden)} forbidden columns in dataset (expected; will be excluded):")
    for c in found_forbidden:
        print(f"    [FORBIDDEN] {c}")

    # -- Null summary --------------------------------------------------------
    print_section("9. Null Value Summary (columns with any nulls)")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if nulls.empty:
        print("  [OK] No null values found.")
    else:
        for col, cnt in nulls.items():
            print(f"  {col:<45}  {cnt:>6,} nulls  ({cnt/len(df)*100:.2f}%)")

    # -- Date range ----------------------------------------------------------
    print_section("10. Order Date Range (for chronological split)")
    if "order date (DateOrders)" in df.columns:
        dates = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
        print(f"  Earliest : {dates.min()}")
        print(f"  Latest   : {dates.max()}")
        print(f"  Span     : {(dates.max() - dates.min()).days} days")
        print(f"  Nulls    : {dates.isna().sum()}")
    else:
        print("  [WARN] 'order date (DateOrders)' NOT FOUND.")


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------

CITATION = """\
Citation:
  Constante, F., Silva, F., Pereira, A. (2019).
  "DataCo Smart Supply Chain for Big Data Analysis."
  Mendeley Data, V5.
  DOI: 10.17632/8gx2fvg2k6.5
  Licence: CC BY 4.0
  Source: https://data.mendeley.com/datasets/8gx2fvg2k6/5
"""


def write_provenance(filepath: Path, csv_path: Path, sha256: str, size_bytes: int) -> None:
    """Write SHA-256 hash, file size, and citation to a provenance text file."""
    content = (
        "RTO Shield AI - Model 1  |  Data Provenance Record\n"
        + ("=" * 60)
        + "\n\n"
        f"File     : {csv_path.name}\n"
        f"Path     : {csv_path.resolve()}\n"
        f"Size     : {size_bytes:,} bytes  ({human_bytes(size_bytes)})\n"
        f"SHA-256  : {sha256}\n\n"
        + CITATION
    )
    filepath.write_text(content, encoding="utf-8")
    print(f"\n  Provenance record saved → {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Verify raw file exists ----------------------------------------------
    if not RAW_CSV.exists():
        print(f"ERROR: Raw CSV not found at {RAW_CSV}")
        print("Please place DataCoSupplyChainDataset.csv in data/raw/ and re-run.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  RTO Shield AI - Model 1  |  Step 1: Data Inspection")
    print("=" * 70)

    # -- Provenance hash (before loading to memory) --------------------------
    print_section("0. File Provenance")
    file_size = RAW_CSV.stat().st_size
    print(f"  Computing SHA-256 hash of {RAW_CSV.name} ...")
    sha256 = sha256_of_file(RAW_CSV)
    print(f"  File size : {file_size:,} bytes  ({human_bytes(file_size)})")
    print(f"  SHA-256   : {sha256}")
    write_provenance(PROVENANCE, RAW_CSV, sha256, file_size)

    # -- Load CSV ------------------------------------------------------------
    print(f"\n  Loading CSV (encoding=latin-1) ...")
    df = load_csv(RAW_CSV)
    print(f"  Loaded successfully.")

    # -- Run inspection ------------------------------------------------------
    inspect(df)

    print("\n" + "=" * 70)
    print("  Inspection complete. No data was modified.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

