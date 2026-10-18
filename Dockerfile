FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=10000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cloud_server.py analysis_python.py ./
COPY web ./web
RUN useradd --create-home --uid 10001 appuser
USER 10001:10001
EXPOSE 10000
CMD ["python", "-u", "cloud_server.py"]
