"""
run_silver_renamu.py

Ejecutar desde la raíz del proyecto:

    python run_silver_renamu.py

Lee Bronze Parquet, aplica silver_rules_renamu.yaml y genera:
- data/silver/renamu/municipalidades_2022/municipalidades_2022_silver.parquet
- reports/silver/renamu/renamu_silver_summary.csv
- reports/silver/renamu/renamu_silver_detail.csv
- reports/silver/renamu/renamu_silver_data_dictionary.csv
- reports/silver/renamu/renamu_silver_dashboard.html
- reports/silver/renamu/municipalidades_2022_silver.html
- data/audit/silver/.../*.json
- logs/pipeline_YYYYMMDD.log
"""

from app.pipelines.silver.silver_pipeline import run_silver_pipeline

if __name__ == "__main__":
    outputs = run_silver_pipeline(source="renamu")
    print("\nTransformación Silver terminada. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")