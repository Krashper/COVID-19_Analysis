FROM apache/airflow:2.9.1 

RUN python -m pip install --upgrade pip
WORKDIR /opt/airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt