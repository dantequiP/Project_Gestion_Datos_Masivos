from app.pipelines.silver.silver_pipeline import run_silver_pipeline


if __name__ == "__main__":
    outputs = run_silver_pipeline(source="sismepre")
    print("\nTransformación Silver terminada. Archivos generados:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
