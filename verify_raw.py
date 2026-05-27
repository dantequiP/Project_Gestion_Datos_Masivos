"""
verify_raw.py — Script independiente para verificar los Parquet RAW generados.

Uso:
    python verify_raw.py
    python verify_raw.py --path data/bronze/siaf
    python verify_raw.py --path data/bronze/sismepre --show-schema
    python verify_raw.py --path data/bronze --show-sample 5
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def verify(
    bronze_path: Path,
    show_schema: bool = False,
    show_sample: int = 0,
) -> None:
    parquet_files = sorted(bronze_path.rglob("*.parquet"))

    if not parquet_files:
        print(f"\n⚠  No se encontraron archivos Parquet en: {bronze_path}\n")
        return

    print(f"\n{'━'*65}")
    print(f"  VERIFICACIÓN RAW — {bronze_path}")
    print(f"  Total archivos Parquet: {len(parquet_files)}")
    print(f"{'━'*65}\n")

    grand_rows = 0
    grand_bytes = 0
    errors = 0

    for pq_file in parquet_files:
        try:
            table = pq.read_table(pq_file)
            size_kb = pq_file.stat().st_size / 1024
            grand_rows += table.num_rows
            grand_bytes += pq_file.stat().st_size

            print(f"  ✓ {pq_file.relative_to(bronze_path)}")
            print(f"    rows={table.num_rows:,}  cols={table.num_columns}  size={size_kb:.1f} KB")

            if show_schema:
                print(f"    Schema:")
                for field in table.schema:
                    print(f"      {field.name}: {field.type}")

            if show_sample > 0:
                df = table.to_pandas()
                print(f"    Primeras {show_sample} filas:")
                with pd.option_context("display.max_columns", None, "display.width", 120):
                    print(df.head(show_sample).to_string(index=False))

            print()

        except Exception as exc:
            print(f"  ✗ {pq_file} — ERROR: {exc}\n")
            errors += 1

    print(f"{'━'*65}")
    print(f"  RESUMEN TOTAL")
    print(f"  Archivos:  {len(parquet_files)}")
    print(f"  Filas:     {grand_rows:,}")
    print(f"  Tamaño:    {grand_bytes / (1024*1024):.2f} MB")
    if errors:
        print(f"  Errores:   {errors}")
    print(f"{'━'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica los Parquet RAW de la capa Bronze")
    parser.add_argument("--path", type=str, default="data/bronze", help="Ruta a verificar")
    parser.add_argument("--show-schema", action="store_true", help="Muestra el schema de cada archivo")
    parser.add_argument("--show-sample", type=int, default=0, metavar="N", help="Muestra las primeras N filas")
    args = parser.parse_args()

    verify(
        bronze_path=Path(args.path),
        show_schema=args.show_schema,
        show_sample=args.show_sample,
    )
