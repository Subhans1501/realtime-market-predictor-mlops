from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'F23-6109',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'market_movement_prediction_pipeline',
    default_args=default_args,
    description='Automated pipeline for data ingestion and sequential model training',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    ingest_data_task = BashOperator(
        task_id='ingest_market_data',
        bash_command='python src/data/ingestion.py',
    )

    train_model_task = BashOperator(
        task_id='train_deep_learning_models',
        bash_command='python src/models/lstm_gru.py',
    )

    ingest_data_task >> train_model_task