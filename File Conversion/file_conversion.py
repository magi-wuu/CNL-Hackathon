#Simple program that converts mdb files to csv file

#code was made with ai claude
#
"""
mdb_to_csv.py - Convert a Microsoft Access (.mdb / .accdb) file to CSV.

Usage from another Python file:

    from mdb_to_csv import mdb_to_csv

    csv_files = mdb_to_csv("data/mydb.mdb")              # all tables
    # -> {"Customers": Path(".../Customers.csv"), "Orders": Path(".../Orders.csv")}

    csv_files = mdb_to_csv("data/mydb.mdb", table="Customers")
    customers_csv = csv_files["Customers"]

Backends (chosen automatically):
  * mdbtools  (Linux/macOS)  ->  sudo apt install mdbtools   /   brew install mdbtools
  * pyodbc    (Windows)      ->  pip install pyodbc  (needs the Microsoft Access ODBC driver)
"""

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# mdbtools backend
# --------------------------------------------------------------------------- #
def _list_tables_mdbtools(mdb_path: Path) -> List[str]:
    result = subprocess.run(
        ["mdb-tables", "-1", str(mdb_path)],
        capture_output=True, text=True, check=True,
    )
    return [t.strip() for t in result.stdout.splitlines() if t.strip()]


def _export_table_mdbtools(mdb_path: Path, table: str, out_file: Path) -> None:
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        subprocess.run(
            ["mdb-export", str(mdb_path), table],
            stdout=f, stderr=subprocess.PIPE, text=True, check=True,
        )


# --------------------------------------------------------------------------- #
# pyodbc backend
# --------------------------------------------------------------------------- #
def _connect_pyodbc(mdb_path: Path):
    import pyodbc  # imported lazily so mdbtools users don't need it

    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        rf"DBQ={mdb_path};"
    )
    return pyodbc.connect(conn_str)


def _list_tables_pyodbc(conn) -> List[str]:
    cursor = conn.cursor()
    return [row.table_name for row in cursor.tables(tableType="TABLE")]


def _export_table_pyodbc(conn, table: str, out_file: Path) -> None:
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM [{table}]")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([col[0] for col in cursor.description])
        for row in cursor:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def mdb_to_csv(
    mdb_file: PathLike,
    output_dir: Optional[PathLike] = None,
    table: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Convert tables in an Access database to CSV files.

    Args:
        mdb_file:   Path to the .mdb/.accdb file.
        output_dir: Where to write the CSVs. Defaults to a folder next to the
                    database named "<db name>_csv".
        table:      Convert only this table. Defaults to every table.

    Returns:
        Dict mapping table name -> Path of the CSV file written.

    Raises:
        FileNotFoundError: if the database file doesn't exist.
        ValueError:        if `table` isn't in the database.
        RuntimeError:      if no supported backend is available.
    """
    mdb_path = Path(mdb_file).expanduser().resolve()
    if not mdb_path.is_file():
        raise FileNotFoundError(f"Database file not found: {mdb_path}")

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else mdb_path.parent / f"{mdb_path.stem}_csv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    use_mdbtools = shutil.which("mdb-export") and shutil.which("mdb-tables")
    conn = None

    if use_mdbtools:
        tables = _list_tables_mdbtools(mdb_path)
    else:
        try:
            conn = _connect_pyodbc(mdb_path)
        except ImportError:
            raise RuntimeError(
                "No backend available. Install mdbtools (Linux/macOS) "
                "or pyodbc + the Access ODBC driver (Windows)."
            )
        tables = _list_tables_pyodbc(conn)

    if table is not None:
        if table not in tables:
            raise ValueError(f"Table '{table}' not found. Available: {tables}")
        tables = [table]

    results: Dict[str, Path] = {}
    try:
        for name in tables:
            safe_name = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in name)
            out_file = out_dir / f"{safe_name}.csv"
            if use_mdbtools:
                _export_table_mdbtools(mdb_path, name, out_file)
            else:
                _export_table_pyodbc(conn, name, out_file)
            results[name] = out_file
    finally:
        if conn is not None:
            conn.close()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python mdb_to_csv.py <file.mdb> [output_dir] [table]")

    files = mdb_to_csv(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None,
    )
    for tbl, path in files.items():
        print(f"{tbl} -> {path}")