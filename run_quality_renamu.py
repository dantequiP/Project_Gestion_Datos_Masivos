"""
run_quality_renamu.py

Ejecutar desde la raíz del proyecto:

    python run_quality_renamu.py

No transforma datos. Solo genera:
- reports/quality/renamu/renamu_quality_detail.csv
- reports/quality/renamu/renamu_quality_summary.csv
- reports/quality/renamu/renamu_quality_failed_samples.csv
- reports/quality/renamu/renamu_quality_dashboard.html
- reports/quality/renamu/municipalidades_2022_quality.html
- data/quality/renamu/*.parquet
- data/audit/quality/.../*.json
- logs/pipeline_YYYYMMDD.log
"""

from app.pipelines.quality.quality_pipeline import run_quality_pipeline

if __name__ == "__main__":
    outputs = run_quality_pipeline(source="renamu")
    print("\nCalidad de datos terminada. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")