FROM apache/airflow:3.1.7

USER root
RUN mkdir -p /opt/airflow/dags
COPY --chown=airflow:root python_doc_service/airflow/dags /opt/airflow/dags

USER airflow
