# Proyecto Gestión de Datos Masivos

## Caso: Presupuesto y Ejecución de Ingresos Municipales

Este proyecto corresponde al curso de **Gestión de Datos Masivos** y tiene como objetivo realizar el análisis de datos para la toma de decisiones sobre el **Presupuesto y Ejecución de Ingresos del MEF**, enfocado en municipalidades del Perú.

El proyecto trabaja con fuentes de datos abiertas proporcionadas por el docente, aplicando una arquitectura de datos tipo **Medallion Architecture** para organizar el flujo de procesamiento de datos desde la ingesta hasta las etapas posteriores de profiling, calidad de datos, transformación y análisis.

---

## Fuentes de datos

Las fuentes consideradas para el proyecto son:

1. **Presupuesto y Ejecución de Ingreso - MEF / SIAF**  
   https://datosabiertos.mef.gob.pe/dataset/presupuesto-y-ejecucion-de-ingreso

2. **Seguimiento de la Meta del Impuesto Predial - MEF / SISMEPRE**  
   https://datosabiertos.mef.gob.pe/dataset/seguimiento-de-la-meta-del-impuesto-predial

3. **Registro Nacional de Municipalidades - RENAMU 2022 / INEI**  
   https://www.datosabiertos.gob.pe/dataset/registro-nacional-de-municipalidades-renamu-2022-instituto-nacional-de-estad%C3%ADstica-e

4. **Dashboard de referencia**  
   https://app.powerbi.com/view?r=eyJrIjoiYjc3NTkxYzUtYmE3OS00ZmNkLTg4ZDQtNjM2NjZlMWNkZTAwIiwidCI6ImQ0NDUyNmQzLTQxNDMtNDU5YS1hNjFhLTcxYWE3YWZkMjIwMiJ9&pageName=ReportSection

---

## Objetivos del proyecto

- Construir una arquitectura de datos basada en **Medallion Architecture**.
- Desarrollar pipelines de ingesta de datos desde fuentes abiertas.
- Convertir los datasets descargados a formato **Parquet** para mejorar el rendimiento y la eficiencia en el análisis.
- Preparar la capa **Bronze** como base para el profiling y control de calidad de datos.
- Documentar el proceso de calidad de datos.
- Construir las capas **Silver** y **Gold** en etapas posteriores.
- Desarrollar dashboards para la toma de decisiones usando Power BI u otra herramienta de visualización.

---

## Arquitectura del proyecto

El proyecto sigue una estructura basada en capas:

```text
data/
├── bronze/   # Datos crudos descargados y convertidos a Parquet
├── silver/   # Datos limpios, integrados y validados
├── gold/     # Datos preparados para análisis, KPIs y dashboards
├── audit/    # Evidencias y auditoría de la ingesta
└── quality/  # Resultados o insumos relacionados con calidad de datos
```

### Capa Bronze

La capa `bronze` contiene los datos descargados desde las fuentes originales y convertidos a formato `.parquet`.

Esta capa se incluye en el repositorio para que los integrantes del equipo puedan avanzar directamente con las tareas de:

- Profiling de datos.
- Validación de calidad.
- Análisis exploratorio.
- Preparación de la capa Silver.

No es necesario volver a descargar las fuentes desde cero para empezar con el análisis.

---

## Estructura general del proyecto

```text
PROYECTO DATOS MASIVOS/
├── app/                  # Código principal del proyecto
├── data/                 # Datos organizados por capas
│   ├── audit/            # Auditoría de ingesta
│   ├── bronze/           # Datos descargados y convertidos a Parquet
│   ├── gold/             # Datos finales para análisis y dashboards
│   ├── quality/          # Resultados o insumos de calidad
│   └── silver/           # Datos limpios e integrados
├── main.py               # Archivo principal de ejecución
├── verify_raw.py         # Script para verificar archivos de datos
├── requirements.txt      # Dependencias del proyecto
├── .gitignore            # Archivos y carpetas ignoradas por Git
└── README.md             # Documentación del proyecto
```

---

## Requisitos previos

Para ejecutar el proyecto se necesita tener instalado:

- Python 3.10 o superior.
- Git.
- Visual Studio Code u otro editor de código.
- Power BI Desktop, si se trabajarán los dashboards.

---

# Guía para clonar y ejecutar el proyecto

Esta sección explica los pasos para descargar/clonar el proyecto desde GitHub y preparar el entorno de trabajo local.

---

## 1. Clonar el repositorio

Abrir Git Bash o la terminal de Visual Studio Code y ejecutar:

```bash
git clone https://github.com/dantequiP/Project_Gestion_Datos_Masivos.git
```

---

## 2. Entrar a la carpeta del proyecto

```bash
cd Project_Gestion_Datos_Masivos
```

---

## 3. Abrir el proyecto en Visual Studio Code

```bash
code .
```

---

## 4. Crear el entorno virtual de Python

Dentro de la carpeta del proyecto, ejecutar:

```bash
python -m venv .venv
```

Este comando crea una carpeta llamada `.venv`, donde se instalarán las dependencias del proyecto.

---

## 5. Activar el entorno virtual

### Si usas Git Bash:

```bash
source .venv/Scripts/activate
```

### Si usas PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Si se activó correctamente, la terminal mostrará algo parecido a:

```bash
(.venv)
```

---

## 6. Instalar dependencias

Con el entorno virtual activado, ejecutar:

```bash
pip install -r requirements.txt
```

Esto instalará las librerías necesarias del proyecto, como `pandas`, `pyarrow`, `requests`, entre otras.

---

## 7. Verificar que los datos estén disponibles

El proyecto ya incluye la capa **Bronze** con archivos en formato `.parquet`.

Para verificar los datos, ejecutar:

```bash
python verify_raw.py
```

Este script permite revisar los archivos descargados y convertidos a formato `.parquet`.

---

## 8. Flujo de trabajo recomendado para el equipo

Cada integrante debería trabajar en una rama propia para evitar conflictos en la rama principal.

Ejemplo:

```bash
git checkout -b feature/profiling-nombre
```

Después de realizar cambios:

```bash
git add .
git commit -m "Add data profiling analysis"
git push origin feature/profiling-nombre
```

Luego se puede crear un **Pull Request** en GitHub para integrar los cambios a la rama principal.

---

## 9. Archivos que no deben subirse

Por buenas prácticas, este proyecto no incluye archivos locales o generados automáticamente como:

```text
.venv/
.env
__pycache__/
logs/
*.pyc
```

Estos archivos no son necesarios para clonar y ejecutar el proyecto, ya que cada integrante puede generar su propio entorno virtual e instalar las dependencias usando `requirements.txt`.

---

## 10. Archivos importantes que sí se incluyen

En este proyecto sí se incluyen los archivos necesarios para continuar con profiling y calidad de datos:

```text
app/
data/bronze/
data/audit/
data/quality/
data/silver/
data/gold/
main.py
verify_raw.py
requirements.txt
README.md
.gitignore
```

La carpeta más importante para continuar el trabajo es:

```text
data/bronze/
```

Ahí se encuentran los archivos `.parquet` ya generados.

---

## Flujo general del proyecto

```text
Fuentes abiertas
        ↓
Ingesta mediante pipeline
        ↓
Conversión a formato Parquet
        ↓
Capa Bronze
        ↓
Profiling de datos
        ↓
Control de calidad de datos
        ↓
Capa Silver
        ↓
Capa Gold
        ↓
Dashboards y análisis para toma de decisiones
```

---

## Problemas comunes

### Error: `python` no se reconoce como comando

Verificar que Python esté instalado y agregado al PATH.

También se puede probar:

```bash
py --version
```

Y crear el entorno con:

```bash
py -m venv .venv
```

---

### Error al activar el entorno en PowerShell

Si aparece un error de permisos, ejecutar PowerShell como administrador y usar:

```powershell
Set-ExecutionPolicy RemoteSigned
```

Luego volver a intentar:

```powershell
.venv\Scripts\Activate.ps1
```

---

### Error instalando `pyarrow`

Actualizar `pip`:

```bash
python -m pip install --upgrade pip
```

Luego volver a instalar:

```bash
pip install -r requirements.txt
```

---

## Resumen rápido para clonar y ejecutar

```bash
git clone https://github.com/dantequiP/Project_Gestion_Datos_Masivos.git
cd Project_Gestion_Datos_Masivos
code .
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python verify_raw.py
```

---

## Estado actual del proyecto

Actualmente el proyecto cuenta con:

- Pipeline inicial de descarga de datos.
- Conversión de fuentes a formato `.parquet`.
- Capa Bronze generada.
- Archivos base para verificación.
- Estructura inicial para continuar con profiling y calidad de datos.

---

## Próximos pasos

1. Realizar profiling de los datasets.
2. Definir reglas de calidad de datos.
3. Documentar problemas encontrados.
4. Construir la capa Silver.
5. Preparar indicadores para dashboards.
6. Desarrollar dashboards para toma de decisiones.



