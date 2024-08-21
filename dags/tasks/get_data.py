import pandas as pd
from datetime import datetime
import logging


def get_date(value):
    try:
        if isinstance(value, datetime):
            day = value.day
            month = value.month
            year = value.year
            return datetime(year, day, month)
        
        return datetime.strptime(str(value), "%m/%d/%y")
    
    except Exception as e:
        logging.error("Ошибка во время преобразования дат: ", e)
        return datetime.strptime(str(value), "%d/%m/%y")


def save_data(data: pd.DataFrame, path: str):
    data.to_csv(path)
    return


def get_data(path: str = "", file_name: str = ""):
    data = pd.read_excel(path)

    data = data.iloc[1:, :]

    data["Country/Region"] = data["Country/Region"].apply(get_date)

    index = data[data["Country/Region"] == datetime(2022, 1, 1)].index.to_list()[0]

    data = data[index:]

    data = data.reset_index(drop=True)

    data = data.rename(columns={"Country/Region": "Date"})

    data = data.set_index("Date")

    save_data(data=data, path=f"dags/data/{file_name}")

    return