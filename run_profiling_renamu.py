"""
run_profiling_renamu.py

Ejecutar desde la raíz del proyecto:

    python run_profiling_renamu.py

No transforma datos. Solo genera:
- reports/profiling/renamu/renamu_profile_detail.csv
- reports/profiling/renamu/renamu_profile_summary.csv
- reports/profiling/renamu/renamu_profile.html
- reports/profiling/renamu/renamu_profile_dashboard.html
- data/audit/profiling/.../*.json
- logs/pipeline_YYYYMMDD.log
"""

from app.pipelines.profiling.profiling_pipeline import run_profiling_pipeline

if __name__ == "__main__":
    outputs = run_profiling_pipeline(source="renamu")
    print("\nProfiling terminado. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")