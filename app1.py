# PART 2 — Flask API to load and predict

from flask import Flask, request, jsonify
import pandas as pd
import os
import re
import joblib

app = Flask(__name__)

EXCEL_FILE = "BDC Information Sheet.xlsx"
SHEET_NAME = "Board Grades"
MODEL_DIR = "models"

df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()
df['Name'] = df['Name'].astype(str).str.strip()

def convert(val):
    if 'K' in val:
        return int(float(val.replace('K', '').replace('+', '')) * 1000)
    elif '+' in val:
        return int(val.replace('+', '').replace('K', '')) * 1000
    return int(val)

# Build range map with cost & setup columns
range_map = {}
setup_cost_map = {}

for i, col in enumerate(df.columns):
    match = re.search(r"\(([\dK\+]+)-?([\dK]*)\s*SF\)", col)
    if match:
        start = convert(match.group(1))
        end = convert(match.group(2)) if match.group(2) else int(start * 1.5)
        range_map[(start, end)] = col

        # Find setup cost column
        next_col = df.columns[i + 1] if i + 1 < len(df.columns) else None
        if next_col and 'Setup' in next_col:
            setup_cost_map[(start, end)] = next_col

@app.route("/")
def home():
    return "Use → /user?board=BOARD_NAME&qty=QUANTITY"

@app.route("/user")
def user():
    board = request.args.get("board", "").strip().upper()
    qty_str = request.args.get("qty", "")

    try:
        qty = float(qty_str)
        board_row = df[df['Name'].str.upper() == board]
        if board_row.empty:
            return jsonify({"error": "Board not found."})

        for (start, end), col in range_map.items():
            if start <= qty < end:
                safe_board = board.replace(" ", "_").replace("/", "_")
                model_file = f"{safe_board}_{start}_{end}.pkl"
                model_path = os.path.join(MODEL_DIR, model_file)

                if not os.path.exists(model_path):
                    return jsonify({"error": f"Model not found for {board} {start}-{end}"})

                model = joblib.load(model_path)
                predicted_cost = model.predict([[qty]])[0]

                setup_col = setup_cost_map.get((start, end))
                setup_cost = None
                if setup_col:
                    setup_cost = board_row.iloc[0][setup_col]

                return jsonify({
                    "Board": board,
                    "Quantity": qty,
                    "Range": f"({start}-{end} SF)",
                    "Predicted Cost ($/MSF)": round(predicted_cost, 4),
                    "Setup Cost": float(setup_cost) if setup_cost is not None else None
                })

        return jsonify({"error": "No matching range found."})
    except ValueError:
        return jsonify({"error": "Invalid quantity input."})

if __name__ == "__main__":
    from urllib.parse import quote
    board = "200BDBLK"
    print(f"API running at http://127.0.0.1:5000/user?board={quote(board)}&qty=12496")
    app.run(debug=True)
