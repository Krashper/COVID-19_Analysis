import pandas as pd
import matplotlib.pyplot as plt
import logging
from tasks.classes.TSForecast import TimeSeriesForecast
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime 
from datetime import timedelta 
import os
import pickle


def get_dates(data, length: int):
    begin_date = data.index[-1]

    if length > 0:
        end_date = begin_date + timedelta(days=length)

    dates = pd.date_range(begin_date, end_date)[1:]

    return dates


def save_model(model, country: str = "", output_file_name: str = "", bucket_name: str = ""):
    try:
        dir_path = f"dags/data/models/{country}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        file_path = dir_path + "/" + output_file_name

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        hook = S3Hook(aws_conn_id="minio_conn")

        hook.load_file(
            filename=file_path,
            key=f'models/{country}/{output_file_name}',  # указываем путь к папке
            bucket_name=bucket_name,
            replace=True
        )
    
    except Exception as e:
        logging.error("Ошибка при сохранении модели: ", e)


def save_dataset(dataset, output_file_name: str = "", bucket_name: str = ""):
    try:
        file_path = f"dags/data/{output_file_name}"
        dataset.to_csv(file_path, index=False)

        hook = S3Hook(aws_conn_id="minio_conn")

        hook.load_file(
            filename=file_path,
            key=f'datasets/{output_file_name}',  # указываем путь к папке
            bucket_name=bucket_name,
            replace=True
        )

    except Exception as e:
        logging.error("Ошибка при сохранении датасета: ", e)


def predict_data(
        input_file_name: str, output_file_name: str, 
        bucket_name: str, pred_days: int = 1, data_type: str = ""):

    dataset = pd.read_csv(f"dags/data/{input_file_name}", index_col=0)

    dataset.index = pd.to_datetime(dataset.index)

    final_dataset = pd.DataFrame(columns=["Date", "Country", "Total_cases", "is_Pred"])

    dates = dataset.index.tolist() +  get_dates(dataset, pred_days).tolist()

    model = None


    for country in dataset.columns:
        print("-" * 50)
        print(country)
        
        country_dataset = dataset[country].astype(float)

        forecast = TimeSeriesForecast(country_dataset)

        forecast.train_test_split(look_back=30)

        forecast.create_model(model)

        forecast.fit_model()

        score = forecast.get_score()

        print(f"Train RMSE: {score[0]}")
        print(f"Test RMSE: {score[1]}")

        model = forecast.model

        forecast.get_future_values(num_predictions=pred_days)

        country_final_dataset = pd.DataFrame({
            "Date": dates,
            "Country": [country] * len(dates),
            "Total_cases": country_dataset.values.tolist() + forecast.predicted_values,
            "is_Pred": [0] * (len(dates) - pred_days) + [1] * pred_days
        })

        final_dataset = pd.concat([final_dataset, country_final_dataset], ignore_index=True)

        save_model(forecast.scaler, country, f"{data_type}_scaler.pkl", bucket_name)
        save_model(forecast.model, country, f"{data_type}_TS_model.pkl", bucket_name)

    save_dataset(final_dataset, output_file_name, bucket_name)