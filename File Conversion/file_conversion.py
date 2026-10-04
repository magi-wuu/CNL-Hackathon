#Simple program that converts mdb files to csv file

#code was made with ai claude
# Link to the Chat log: https://claude.ai/share/bb29027b-5d89-4973-bac9-5490b2973ee8
"""
file_conversion.py - Convert Microsoft Access (.mdb / .accdb) files to CSV.

Convert ONE file:
    from file_conversion import mdb_to_csv
    files = mdb_to_csv("test.mdb")                  # {table: Path to csv}

Convert a WHOLE FOLDER (keeps the same folder structure and folder names):
    from file_conversion import convert_folder
    converted, failed = convert_folder("MyData")    # writes MyData_csv next to it

Command line:
    py file_conversion.py test.mdb                  # one file
    py file_conversion.py MyData                    # whole folder -> MyData_csv
    py file_conversion.py MyData C:\\out\\folder    # whole folder -> chosen output

Backends (tried in this order):
  1. mdbtools      (Linux/macOS)
  2. pyodbc        (Windows, needs the Access ODBC driver)
  3. access-parser (any OS, no driver)  ->  pip install access-parser
"""

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]
EXTENSIONS = (".mdb", ".accdb")


# ---------------- mdbtools ----------------
def _list_tables_mdbtools(mdb_path: Path) -> List[str]:
    result = subprocess.run(["mdb-tables", "-1", str(mdb_path)],
                            capture_output=True, text=True, check=True)
    return [t.strip() for t in result.stdout.splitlines() if t.strip()]


def _export_table_mdbtools(mdb_path: Path, table: str, out_file: Path) -> None:
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        subprocess.run(["mdb-export", str(mdb_path), table],
                       stdout=f, stderr=subprocess.PIPE, text=True, check=True)


# ---------------- pyodbc ----------------
def _connect_pyodbc(mdb_path: Path):
    import pyodbc
    conn_str = (r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                rf"DBQ={mdb_path};")
    return pyodbc.connect(conn_str)


def _list_tables_pyodbc(conn) -> List[str]:
    return [r.table_name for r in conn.cursor().tables(tableType="TABLE")]


def _export_table_pyodbc(conn, table: str, out_file: Path) -> None:
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM [{table}]")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c[0] for c in cursor.description])
        for row in cursor:
            writer.writerow(row)


# ---------------- access-parser ----------------
def _open_access_parser(mdb_path: Path):
    from access_parser import AccessParser
    return AccessParser(str(mdb_path))


def _list_tables_access_parser(db) -> List[str]:
    return [t for t in db.catalog if not str(t).startswith("MSys")]


def _export_table_access_parser(db, table: str, out_file: Path) -> None:
    data = db.parse_table(table)          # {column: [values...]}
    cols = list(data.keys())
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        if cols:
            for row in zip(*(data[c] for c in cols)):
                writer.writerow(row)


# ---------------- shared helpers ----------------
def _open_backend(mdb_path: Path):
    """Return (backend_name, handle, table_names) using the first backend that works."""
    errors = []

    if shutil.which("mdb-export") and shutil.which("mdb-tables"):
        return "mdbtools", None, _list_tables_mdbtools(mdb_path)

    try:
        conn = _connect_pyodbc(mdb_path)
        return "pyodbc", conn, _list_tables_pyodbc(conn)
    except Exception as e:
        errors.append(f"pyodbc: {e}")

    try:
        db = _open_access_parser(mdb_path)
        return "access-parser", db, _list_tables_access_parser(db)
    except Exception as e:
        errors.append(f"access-parser: {e}")

    raise RuntimeError(
        "No working backend. Easiest fix: pip install access-parser\n"
        "Details:\n  " + "\n  ".join(errors)
    )


def _close_backend(backend: str, handle) -> None:
    if backend == "pyodbc" and handle is not None:
        handle.close()


def _export(backend: str, handle, mdb_path: Path, table: str, out_file: Path) -> None:
    if backend == "mdbtools":
        _export_table_mdbtools(mdb_path, table, out_file)
    elif backend == "pyodbc":
        _export_table_pyodbc(handle, table, out_file)
    else:
        _export_table_access_parser(handle, table, out_file)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in name)


# ---------------- convert ONE file ----------------
def mdb_to_csv(
    mdb_file: PathLike,
    output_dir: Optional[PathLike] = None,
    table: Optional[str] = None,
) -> Dict[str, Path]:
    """Convert every table (or just `table`) of one Access file. Returns {table: csv Path}."""
    mdb_path = Path(mdb_file).expanduser().resolve()
    if not mdb_path.is_file():
        raise FileNotFoundError(f"Database file not found: {mdb_path}")

    out_dir = (Path(output_dir).expanduser().resolve() if output_dir
               else mdb_path.parent / f"{mdb_path.stem}_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend, handle, tables = _open_backend(mdb_path)
    if table is not None:
        if table not in tables:
            _close_backend(backend, handle)
            raise ValueError(f"Table '{table}' not found. Available: {tables}")
        tables = [table]

    results: Dict[str, Path] = {}
    try:
        for name in tables:
            out_file = out_dir / f"{_safe(name)}.csv"
            _export(backend, handle, mdb_path, name, out_file)
            results[name] = out_file
    finally:
        _close_backend(backend, handle)
    return results


# ---------------- convert a WHOLE FOLDER ----------------
def convert_folder(
    input_dir: PathLike,
    output_dir: Optional[PathLike] = None,
) -> Tuple[Dict[Path, List[Path]], List[Tuple[Path, str]]]:
    """
    Convert every .mdb/.accdb under `input_dir` (including all subfolders) to CSV.

    The output folder has exactly the same folder structure and folder names.
    Each Access file becomes:
        - <name>.csv                if it has ONE table
        - <name>_<table>.csv        for each table, if it has several
    and is placed in the matching subfolder of the output.

    The input folder is never modified. One bad file does not stop the rest.

    Returns (converted, failed):
        converted: {mdb Path: [csv Paths]}
        failed:    [(mdb Path, error message)]
    """
    in_dir = Path(input_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {in_dir}")

    out_dir = (Path(output_dir).expanduser().resolve() if output_dir
               else in_dir.parent / f"{in_dir.name}_csv")
    if out_dir == in_dir:
        raise ValueError("Output folder must be different from the input folder.")

    def inside_output(p: Path) -> bool:
        return out_dir == p or out_dir in p.parents

    # Recreate every folder (even ones with no Access files) so the layout matches.
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in sorted(in_dir.rglob("*")):
        if d.is_dir() and not inside_output(d.resolve()):
            (out_dir / d.relative_to(in_dir)).mkdir(parents=True, exist_ok=True)

    mdbs = sorted(
        p for p in in_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
        and not inside_output(p.resolve())
    )

    converted: Dict[Path, List[Path]] = {}
    failed: List[Tuple[Path, str]] = []
    print(f"Found {len(mdbs)} Access file(s) in {in_dir}")

    for i, mdb in enumerate(mdbs, 1):
        rel = mdb.relative_to(in_dir)
        target = out_dir / rel.parent
        target.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(mdbs)}] {rel} ...", end=" ", flush=True)
        try:
            backend, handle, tables = _open_backend(mdb)
            try:
                paths = []
                for t in tables:
                    fname = f"{mdb.stem}.csv" if len(tables) == 1 else f"{mdb.stem}_{_safe(t)}.csv"
                    out_file = target / fname
                    _export(backend, handle, mdb, t, out_file)
                    paths.append(out_file)
            finally:
                _close_backend(backend, handle)
            converted[mdb] = paths
            print(f"done ({len(paths)} table(s))")
        except Exception as e:
            failed.append((mdb, str(e)))
            print("FAILED")

    print(f"\nFinished: {len(converted)} converted, {len(failed)} failed.")
    print(f"Output folder: {out_dir}")
    for mdb, err in failed:
        print(f"  FAILED: {mdb.relative_to(in_dir)} -> {err.splitlines()[0]}")
    return converted, failed


if __name__ == "__main__":
    import sys

    # Edit this default if you just press Run in your editor:
    target = sys.argv[1] if len(sys.argv) > 1 else "test.mdb"
    second = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        if Path(target).expanduser().is_dir():
            convert_folder(target, second)
        else:
            print(f"Converting {target} ...")
            files = mdb_to_csv(target, second, sys.argv[3] if len(sys.argv) > 3 else None)
            for tbl, path in files.items():
                print(f"  {tbl} -> {path}")
            print(f"Done! Converted {len(files)} table(s).")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)