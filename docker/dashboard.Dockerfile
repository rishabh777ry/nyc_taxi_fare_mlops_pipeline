# ─── Streamlit Dashboard ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    plotly \
    requests

COPY dashboard/ dashboard/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
