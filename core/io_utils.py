# ==============================
# core/io_utils.py
# ==============================
from pathlib import Path
from typing import Union, Any
import pandas as pd
import io
from chardet import detect


def load_any(path_or_file: Union[str, Path, Any]) -> pd.DataFrame:
    """
    Universal loader supporting:
        - CSV
        - Parquet
        - Excel (XLS/XLSX)
        - TXT (single column)
    
    Works for both:
        • Local filesystem path
        • Streamlit UploadedFile object
    """

    # ===== UploadedFile branch =====
    if hasattr(path_or_file, "name") and not isinstance(path_or_file, (str, Path)):
        name = path_or_file.name.lower()

        # ---- CSV ----
        if name.endswith(".csv"):
            raw_bytes = path_or_file.getvalue()
            sample = raw_bytes[:4096]
            enc = detect(sample).get("encoding") or "utf-8"
            text = raw_bytes.decode(enc, errors="ignore")
            return pd.read_csv(io.StringIO(text), low_memory=False)

        # ---- PARQUET ----
        if name.endswith(".parquet"):
            return pd.read_parquet(path_or_file)

        # ---- EXCEL ----
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(path_or_file)

        # ---- TXT ----
        if name.endswith(".txt"):
            raw_bytes = path_or_file.getvalue()
            sample = raw_bytes[:4096]
            enc = detect(sample).get("encoding") or "utf-8"
            text = raw_bytes.decode(enc, errors="ignore")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return pd.DataFrame(lines, columns=["value"])

        raise ValueError(f"Unsupported uploaded file type: {name}")

    # ===== Local path branch =====
    p = Path(path_or_file)
    suffix = p.suffix.lower()

    # ---- CSV ----
    if suffix == ".csv":
        return pd.read_csv(p, low_memory=False)

    # ---- PARQUET ----
    if suffix == ".parquet":
        return pd.read_parquet(p)

    # ---- EXCEL ----
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(p)

    # ---- TXT ----
    if suffix == ".txt":
        with open(p, "rb") as f:
            raw = f.read(4096)
        enc = detect(raw).get("encoding") or "utf-8"
        with open(p, "r", encoding=enc, errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame(lines, columns=["value"])

    raise ValueError(f"Unsupported file format: {p}")