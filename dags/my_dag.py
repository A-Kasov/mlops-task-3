from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 1, 16, 1, 9),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False,
}

with DAG(
    "Red-Wine",
    description="Red-Wine good/bad",
    schedule_interval="*/1 * * * *",
    default_args=args,
    tags=["Red-Wine", "classification"],
) as dag:
    data_download = BashOperator(
        task_id="data_download",
        bash_command="python3 /home/artem/Projects/mlops-task3/scripts/data_download.py",
        dag=dag,
    )
    data_prepare = BashOperator(
        task_id="data_prepare",
        bash_command="python3 /home/artem/Projects/mlops-task3/scripts/data_prepare.py",
        dag=dag,
    )
    model_train = BashOperator(
        task_id="model_train",
        bash_command="python3 /home/artem/Projects/mlops-task3/scripts/model_train.py",
        dag=dag,
    )
    data_download >> data_prepare >> model_train