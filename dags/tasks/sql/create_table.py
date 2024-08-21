def create_table(table_name: str):
    return f'''CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            Date DATE,
            Country VARCHAR,
            Total_cases FLOAT,
            is_Pred BOOLEAN
        )'''