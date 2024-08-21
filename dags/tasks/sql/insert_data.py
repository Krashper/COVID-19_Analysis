import pandas as pd
import logging


def insert_data(file_name: str, table_name: str):
    try:
        dataset = pd.read_csv(f"dags/data/{file_name}")

        print(dataset)
        insert_queries = []
        for _, row in dataset.iterrows():
            values = (row["Date"], row["Country"].replace("'", "''"), row["Total_cases"], bool(row["is_Pred"]))
            insert_queries.append(
                f"INSERT INTO {table_name} (Date, Country, Total_cases, is_Pred) VALUES ('{values[0]}', '{values[1]}', {values[2]}, {values[3]}) ON CONFLICT DO NOTHING")

        return ";\n".join(insert_queries)

    except Exception as e:
        logging.error("Ошибка во время добавления данных в БД: ", e)
        return