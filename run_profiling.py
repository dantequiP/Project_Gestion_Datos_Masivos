
from app.pipelines.profiling.profiling_pipeline import run_profiling_pipeline

if __name__ == "__main__":
    outputs = run_profiling_pipeline(source="sismepre")
    print("\nProfiling terminado. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
