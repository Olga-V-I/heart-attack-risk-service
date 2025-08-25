# app.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Body

ID_COL = "id"
model = joblib.load("best_model.joblib")  
app = FastAPI()

@app.post("/predict")
def predict(csv_path: str = Body(..., embed=False)):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {e}")

    # уберём служебный столбец 
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # приводим имена колонок к snake_case — как при обучении
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    #  проверяем id
    if "id" not in df.columns:
        raise HTTPException(status_code=400, detail="В файле нет колонки 'id'")

    #  приводим gender к 'male'/'female' (в исходнике могли быть 0/1)
    if "gender" in df.columns:
        def clean_gender(val):
            s = str(val).lower().strip()
            if s in ("male", "1.0", "1"): return "male"
            if s in ("female", "0.0", "0"): return "female"
            return None
        df["gender"] = df["gender"].apply(clean_gender)

    try:
        proba = model.predict_proba(df.drop(columns=["id"]))[:, 1]
    except ValueError as e:
        # если всё ещё не совпадают колонки — вернём понятную 400-ошибку
        raise HTTPException(status_code=400, detail=f"Проблема с колонками: {e}")

    items = [{"id": int(i), "prediction": float(p)} for i, p in zip(df["id"], proba)]
    return {"items": items}



