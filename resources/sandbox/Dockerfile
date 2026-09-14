FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends nodejs ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pandas matplotlib pillow pypdf openpyxl
COPY runner.py /opt/sophia/runner.py
ENV PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
USER 65534:65534
WORKDIR /workspace
ENTRYPOINT ["python", "-I", "/opt/sophia/runner.py"]
