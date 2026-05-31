from app.pipelines.quality.quality_pipeline import run_quality_pipeline

if __name__ == "__main__":
    outputs = run_quality_pipeline(source="sismepre")
    print("\nCalidad de datos terminada. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")