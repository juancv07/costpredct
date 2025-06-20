# PART 1 — Train and save models

import pandas as pd
import numpy as np
import os
import re
from sklearn.linear_model import LinearRegression
import joblib

EXCEL_FILE = "BDC Information Sheet.xlsx"
SHEET_NAME = "Board Grades"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()
df['Name'] = df['Name'].astype(str).str.strip()

def convert(val):
    if 'K' in val:
        return int(float(val.replace('K', '').replace('+', '')) * 1000)
    elif '+' in val:
        return int(val.replace('+', '').replace('K', '')) * 1000
    return int(val)

for index, row in df.iterrows():
    board_name = row['Name']
    safe_board = board_name.replace(" ", "_").replace("/", "_")

    for i, col in enumerate(df.columns):
        match = re.search(r"\(([\dK\+]+)-?([\dK]*)\s*SF\)", col)
        if match:
            start = convert(match.group(1))
            end = convert(match.group(2)) if match.group(2) else int(start * 1.5)

            cost = row[col]
            if pd.isnull(cost):
                continue

            # Simulate drop across range
            drop_percent = 0.035
            end_cost = cost * (1 - drop_percent)

            X = np.array([[start], [end]])
            y = np.array([cost, end_cost])

            model = LinearRegression().fit(X, y)

            model_path = os.path.join(SAVE_DIR, f"{safe_board}_{start}_{end}.pkl")
            joblib.dump(model, model_path)

print(" All models trained and saved.")
