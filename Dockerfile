FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY video_orientation_detector.py svod_detector.py svod_bootstrap.py svod_state.py svod_cli.py svod_models.py svod_statistics.py __init__.py reference_orientations.csv ./

RUN useradd --create-home --uid 10001 svod \
    && chown -R svod:svod /app
USER svod

RUN python video_orientation_detector.py --version

ENTRYPOINT ["python", "video_orientation_detector.py"]
