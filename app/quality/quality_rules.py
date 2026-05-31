from pathlib import Path
from typing import Any
import yaml

def load_quality_rules(config_path: Path = Path("app/config/quality_rules_sismepre.yaml")) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de reglas: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("El archivo YAML de reglas no tiene una estructura válida.")
    for section in ["settings", "datasets", "global_rules"]:
        if section not in config:
            raise ValueError(f"Falta la sección obligatoria: {section}")
    return config