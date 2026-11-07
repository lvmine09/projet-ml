import os
import json
import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="Projet Machine Learning", page_icon="🤖")

# ---------------------------------------------------------------------------
# Description des modèles : un dossier models/<folder> par modèle contenant
# model.joblib, encoders.joblib, scaler.joblib, uniques.joblib
#   kind "cat" : variable catégorielle, idx = position dans encoders/uniques
#   kind "num" : variable numérique
# L'ordre des variables = l'ordre des colonnes utilisé à l'entraînement.
# ---------------------------------------------------------------------------
CONFIGS = {
    "Car data (classification)": {
        "folder": "car_data",
        "type": "classification",
        "target": "Venant_Occasion",
        "target_idx": 3,  # position de la cible dans encoders
        "features": [
            {"name": "Marque", "kind": "cat", "idx": 0},
            {"name": "Année", "kind": "num", "default": 2015, "step": 1},
            {"name": "Boite_Vitesse", "kind": "cat", "idx": 1},
            {"name": "Adresse", "kind": "cat", "idx": 2},
            {"name": "Prix", "kind": "num", "default": 8000000, "step": 100000},
        ],
    },
    "Car dataset (régression)": {
        "folder": "car_dataset",
        "type": "regression",
        "target": "Selling_Price",
        "features": [
            {"name": "Kms_Driven", "kind": "num", "default": 50000, "step": 1000},
            {"name": "Present_Price", "kind": "num", "default": 10.0, "step": 0.5},
            {"name": "Fuel_Type", "kind": "cat", "idx": 0},
            {"name": "Seller_Type", "kind": "cat", "idx": 1},
            {"name": "Transmission", "kind": "cat", "idx": 2},
            {"name": "Age", "kind": "num", "default": 5, "step": 1},
        ],
    },
    "Taxi trip (régression)": {
        "folder": "taxi_trip",
        "type": "regression",
        "target": "Trip_Price",
        "suffix": "$",
        "features": [
            {"name": "Trip_Distance_km", "kind": "num", "default": 25.0, "step": 1.0},
            {"name": "Time_of_Day", "kind": "cat", "idx": 0},
            {"name": "Day_of_Week", "kind": "cat", "idx": 1},
            {"name": "Passenger_Count", "kind": "num", "default": 1, "step": 1},
            {"name": "Traffic_Conditions", "kind": "cat", "idx": 2},
            {"name": "Weather", "kind": "cat", "idx": 3},
            {"name": "Base_Fare", "kind": "num", "default": 3.5, "step": 0.1},
            {"name": "Per_Km_Rate", "kind": "num", "default": 1.2, "step": 0.1},
            {"name": "Per_Minute_Rate", "kind": "num", "default": 0.3, "step": 0.05},
            {"name": "Trip_Duration_Minutes", "kind": "num", "default": 60.0, "step": 1.0},
        ],
    },
}

# Téléphone : la description des variables est lue dans models/telephone/config.json
# (fichier créé par le notebook Telephone_Classification_corrige.ipynb)
_tel = os.path.join("models", "telephone", "config.json")
if os.path.exists(_tel):
    with open(_tel, encoding="utf-8") as _f:
        CONFIGS["Téléphone (classification)"] = {
            "folder": "telephone", "type": "classification", **json.load(_f)
        }


@st.cache_resource
def load_artifacts(folder):
    path = os.path.join("models", folder)
    model_file = "model.joblib" if os.path.exists(os.path.join(path, "model.joblib")) else "rf_model.joblib"
    return {
        "model": jb.load(os.path.join(path, model_file)),
        "encoders": jb.load(os.path.join(path, "encoders.joblib")),
        "scaler": jb.load(os.path.join(path, "scaler.joblib")),
        "uniques": jb.load(os.path.join(path, "uniques.joblib")),
    }


def predict_row(cfg, art, values):
    """values : dict {nom_variable: valeur}. Même logique que Pred_func du cours."""
    x_new = []
    for f in cfg["features"]:
        v = values[f["name"]]
        if f["kind"] == "cat":
            v = art["encoders"][f["idx"]].transform([v])[0]
        x_new.append(v)
    x_new = np.array(x_new, dtype=float).reshape(1, -1)  # 2D array
    x_new = art["scaler"].transform(x_new)               # normaliser
    y_pred = art["model"].predict(x_new)[0]              # prédire
    if cfg["type"] == "classification":
        return art["encoders"][cfg["target_idx"]].inverse_transform([int(y_pred)])[0]
    return f"{round(float(y_pred), 2)}{cfg.get('suffix', '')}"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.sidebar.title("Projet Machine Learning")
choix = st.sidebar.selectbox("Choisir un modèle", list(CONFIGS))
cfg = CONFIGS[choix]

st.title(choix)

try:
    art = load_artifacts(cfg["folder"])
except FileNotFoundError:
    st.error(f"Fichiers du modèle introuvables dans models/{cfg['folder']}/ "
             "(model.joblib, encoders.joblib, scaler.joblib, uniques.joblib).")
    st.stop()

tab1, tab2 = st.tabs(["Simple Prédiction", "Prédiction multiple"])

# ----- Interface 1 : une seule prédiction
with tab1:
    values = {}
    for f in cfg["features"]:
        if f["kind"] == "cat":
            values[f["name"]] = st.selectbox(f["name"], art["uniques"][f["idx"]], key=f"{choix}_{f['name']}")
        else:
            values[f["name"]] = st.number_input(f["name"], value=float(f["default"]),
                                                step=float(f["step"]), key=f"{choix}_{f['name']}")
    if st.button("Prédire", key=f"{choix}_btn"):
        st.success(f"{cfg['target']} : {predict_row(cfg, art, values)}")

# ----- Interface 2 : prédictions multiples avec un fichier csv
with tab2:
    noms = [f["name"] for f in cfg["features"]]
    st.write("Le fichier csv doit contenir les colonnes : " + ", ".join(noms))
    fichier = st.file_uploader("Upload a csv file", type="csv", key=f"{choix}_file")
    if fichier is not None:
        df = pd.read_csv(fichier)
        manquantes = [c for c in noms if c not in df.columns]
        if manquantes:
            st.error("Colonnes manquantes dans le fichier : " + ", ".join(manquantes))
        else:
            try:
                df[cfg["target"]] = [predict_row(cfg, art, dict(zip(noms, row)))
                                     for row in df[noms].values]
                st.dataframe(df)
                st.download_button("Download a csv file", df.to_csv(index=False),
                                   "predictions.csv", "text/csv")
            except ValueError as e:
                st.error(f"Une valeur du fichier n'est pas reconnue par le modèle : {e}")
