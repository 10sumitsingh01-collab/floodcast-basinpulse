# Reproducible, isolated environment for the BASINSHIFT entry (BASINSHIFT 11.2, 15.1).
#   docker build -t basinpulse .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" basinpulse            # tests
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" basinpulse \
#       python basinshift_run.py --model src.bs_models:RobustMeanModel --name lgbm_final          # full run
# Data is mounted, never baked into the image (unzip basinshift_caravan_v1.zip into ./data/caravan first).
# No network access is needed at run time.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-m", "pytest", "-q", "tests/test_basinshift_eval.py", "tests/test_bs_models.py", "tests/test_submission_schema.py"]
