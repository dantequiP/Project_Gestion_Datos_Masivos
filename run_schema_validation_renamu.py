"""
run_schema_validation_renamu.py

Ejecutar desde la raíz del proyecto:

    python run_schema_validation_renamu.py

No transforma datos. Solo genera:
- reports/schema_validation/renamu/*.csv
- reports/schema_validation/renamu/*.html
- data/audit/schema_validation/.../*.json
- logs/pipeline_YYYYMMDD.log
"""

from app.pipelines.schema_validation.schema_validation_pipeline import run_schema_validation_pipeline


if __name__ == "__main__":
    outputs = run_schema_validation_pipeline(source="renamu")
    print("\nValidación terminada. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")