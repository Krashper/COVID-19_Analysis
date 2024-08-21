from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from tasks.get_data import get_data
from tasks.create_bucket import create_bucket
from tasks.save_data_to_s3 import save_data_to_s3
from tasks.predict_data import predict_data
from tasks.sql.create_table import create_table
from tasks.sql.insert_data import insert_data
from airflow.models import Variable



default_args = {
    'owner': 'Ruslan Solarev',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 23),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="predict_values_TS_40",
    description="It is a dag which gets data from Kinopoisk API and save them into database",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    default_args=default_args
) as dag:
    
    cases_data_path = Variable.get("cases_data_path")
    deaths_data_path = Variable.get("deaths_data_path")

    pred_days = int(Variable.get("pred_days"))

    bucket_name = "covid19"

    create_bucket_task = PythonOperator(
        task_id="create_s3_bucket",
        python_callable=create_bucket,
        op_kwargs={
            "bucket_name": bucket_name
        }
    )
    

    with TaskGroup("get_cases_dataset") as get_cases_dataset:
        file_name = "cases_dataset.csv"

        get_datasets_task = PythonOperator(
            task_id="get_datasets_from_google",
            python_callable=get_data,
            op_kwargs={
                "path": cases_data_path,
                "file_name": file_name
            }
        )

        save_data_to_s3_task = PythonOperator(
            task_id="save_data_to_s3",
            python_callable=save_data_to_s3,
            op_kwargs={
                "bucket_name": bucket_name,
                "file_name": file_name
            }
        )


        predict_data_task = PythonOperator(
            task_id="predict_data",
            python_callable=predict_data,
            op_kwargs={
                "input_file_name": file_name,
                "output_file_name": f"final_{file_name}",
                "bucket_name": bucket_name,
                "pred_days": pred_days,
                "data_type": "cases"
            }
        )

        create_table_task = PostgresOperator(
            task_id='create_morbidity_table',
            postgres_conn_id="postgres",
            sql=create_table(table_name="Morbidity"),
        )

        insert_data_to_db_task = PostgresOperator(
            task_id="insert_data_to_db",
            postgres_conn_id="postgres",
            sql=insert_data(file_name=f"final_{file_name}", table_name="Morbidity")
        )

        get_datasets_task >> save_data_to_s3_task >> [predict_data_task, create_table_task]
        predict_data_task >> insert_data_to_db_task
        create_table_task >> insert_data_to_db_task


    with TaskGroup("get_deaths_dataset") as get_deaths_dataset:
        file_name = "deaths_dataset.csv"

        get_datasets_task = PythonOperator(
            task_id="get_datasets_from_google",
            python_callable=get_data,
            op_kwargs={
                "path": deaths_data_path,
                "file_name": file_name
            }
        )


        save_data_to_s3_task = PythonOperator(
            task_id="save_data_to_s3",
            python_callable=save_data_to_s3,
            op_kwargs={
                "bucket_name": bucket_name,
                "file_name": file_name
            }
        )


        predict_data_task = PythonOperator(
            task_id="predict_data",
            python_callable=predict_data,
            op_kwargs={
                "input_file_name": file_name,
                "output_file_name": f"final_{file_name}",
                "bucket_name": bucket_name,
                "pred_days": pred_days,
                "data_type": "deaths"
            }
        )
        create_table_task = PostgresOperator(
            task_id='create_deaths_table',
            postgres_conn_id="postgres",
            sql=create_table(table_name="Deaths"),
        )

        insert_data_to_db_task = PostgresOperator(
            task_id="insert_data_to_db",
            postgres_conn_id="postgres",
            sql=insert_data(file_name=f"final_{file_name}", table_name="Deaths")
        )

        get_datasets_task >> save_data_to_s3_task >> [predict_data_task, create_table_task]
        predict_data_task >> insert_data_to_db_task
        create_table_task >> insert_data_to_db_task

    create_bucket_task >> [get_cases_dataset, get_deaths_dataset]