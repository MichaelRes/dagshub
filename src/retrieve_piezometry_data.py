"""
Script to retrieve piezometry data being given a list of stations,
using Hubeau API (https://api.gouv.fr/documentation/api_hubeau_piezometrie)
"""


import os

import pandas as pd
import requests
import tqdm

# Path of csv storing the list of stations with their BSS codes
STATIONS_CSV_PATH = "data/stations_ades_1440_aquiferes.csv"

# Path to folder where to save the piezometry time series of all stations
SAVE_FOLDER = "data/piezometry"

# List of stations for which NOT to retrieve piezometry data
# EXCLUDE_STATIONS = ["saint_felix", "bioule", "les_barthes", "saint_porquier", "tarsac", "verniolles"]
# EXCLUDE_STATIONS = ["Bioule", "Les Barthes", "Saint-Porquier", "Verniolle"]
EXCLUDE_STATIONS = []

# URL of the API to retrieve piezometry data for a given station BSS code
PIEZOMETRY_API_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques.csv"


os.makedirs(SAVE_FOLDER, exist_ok=True)
df_stations = pd.read_csv(STATIONS_CSV_PATH, sep=";")
for idx, (code_bss, nom_commune) in tqdm.tqdm(
        enumerate(
            zip(df_stations["code_bss"],
                df_stations["nom_commune"])
        )
):
    if nom_commune in EXCLUDE_STATIONS:
        continue
    result = requests.get(PIEZOMETRY_API_URL,
                          params={"code_bss": code_bss,
                                  "date_debut_mesure": "2005-01-01",
                                  "date_fin_mesure": "2023-01-01",
                                  "size": 366 * 18})
    save_path = os.path.join(SAVE_FOLDER, f"{code_bss.replace('/', '#')}#{nom_commune}.csv")
    with open(save_path, "w") as file:
        file.write(result.text)
    # if idx > 5:
    #    break
