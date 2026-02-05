import pandas as pd


def parquet_data_processing(df_parquet):
    # Ensure that the inventory is numerical.
    df_parquet["Inventario"] = pd.to_numeric(
        df_parquet["Inventario"],
        errors="coerce"
    )

    # Filter inventory NOT null and >= 1
    df_stock_1 = df_parquet[
        df_parquet["Inventario"].notna() &
        (df_parquet["Inventario"] >= 1)
    ]

    return df_stock_1
