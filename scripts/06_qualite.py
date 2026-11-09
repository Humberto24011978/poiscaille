"""Étape 2 ter : qualité de l'eau des rivières (pollution) et état des peuplements de poissons.

Sources (Hub'Eau, gratuites, sans clé) :
- qualité des cours d'eau (Agences de l'eau) : oxygène dissous, ammonium, nitrites, phosphates, matières organiques
  (DBO5), analysées plusieurs fois par an ; publiées avec 1 à 2 ans de retard ;
- indice poissons rivière (IPR) calculé par l'OFB lors des pêches scientifiques.

Classes de qualité (cadre de la directive-cadre sur l'eau, paramètres physico-chimiques généraux) :
1 très bon, 2 bon, 3 moyen, 4 médiocre, 5 mauvais. La station prend la classe du paramètre le plus déclassant,
calculée sur ses 3 dernières années de mesures (10 % des valeurs les plus basses pour l'oxygène,
10 % des plus hautes pour les autres).

Chaque station vaut pour son morceau de rivière et pour le même cours d'eau à moins de 5 km.

Résultat : data/qualite.csv (une ligne par eau rattachée)
"""
import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd

from commun import DATA, DEPARTEMENTS, get, stations_vers_rivieres

API = "https://hubeau.eaufrance.fr/api"
ANNEES = 7
# code : (nom, sens, seuils entre classes 1|2, 2|3, 3|4, 4|5) ; sens « bas » : plus c'est bas, plus c'est mauvais
PARAMETRES = {
    "1311": ("oxygène", "bas", [8, 6, 4, 3]),
    "1335": ("ammonium", "haut", [0.1, 0.5, 2, 5]),
    "1339": ("nitrites", "haut", [0.1, 0.3, 0.5, 1]),
    "1433": ("phosphates", "haut", [0.1, 0.5, 1, 2]),
    "1313": ("matières organiques", "haut", [3, 6, 10, 25]),
}
MIN_MESURES = 4


def toutes_les_pages(url, params):
    """Suit les liens « next » de Hub'Eau."""
    lignes, r = [], get(url, params, timeout=300).json()
    while True:
        lignes += r.get("data") or []
        if not r.get("next"):
            return lignes
        r = get(r["next"], timeout=300).json()


def analyses():
    an = dt.date.today().year
    champs = "code_station,libelle_station,date_prelevement,code_parametre,resultat,longitude,latitude"
    morceaux = []
    for d in DEPARTEMENTS:
        for a in range(an - ANNEES, an + 1):
            l = toutes_les_pages(f"{API}/v2/qualite_rivieres/analyse_pc", {
                "code_departement": d, "code_parametre": ",".join(PARAMETRES), "fields": champs,
                "date_debut_prelevement": f"{a}-01-01", "date_fin_prelevement": f"{a}-12-31", "size": 5000})
            if l:
                morceaux.append(pd.DataFrame(l))
        print(f"  département {d} : {sum(len(m) for m in morceaux)} analyses au total")
    if not morceaux:
        return pd.DataFrame(columns=champs.split(","))
    df = pd.concat(morceaux, ignore_index=True)
    df["resultat"] = pd.to_numeric(df["resultat"], errors="coerce")
    df["code_parametre"] = df["code_parametre"].astype(str)
    df["date"] = pd.to_datetime(df["date_prelevement"], errors="coerce")
    return df.dropna(subset=["resultat", "date", "longitude", "latitude"])


def classe(valeur, sens, seuils):
    if sens == "bas":
        return 1 + sum(valeur < s for s in seuils)
    return 1 + sum(valeur > s for s in seuils)


def qualite_stations(df):
    """Classe de chaque station (paramètre le plus déclassant sur ses 3 dernières années de mesures)."""
    res = []
    for code, g in df.groupby("code_station"):
        fin = g["date"].max()
        g = g[g["date"] > fin - pd.Timedelta(days=3 * 365)]
        pire, motifs = 0, []
        for p, (nom, sens, seuils) in PARAMETRES.items():
            v = g.loc[g["code_parametre"] == p, "resultat"]
            if len(v) < MIN_MESURES:
                continue
            stat = np.percentile(v, 10 if sens == "bas" else 90)
            c = classe(stat, sens, seuils)
            if c > pire:
                pire, motifs = c, [nom]
            elif c == pire and c >= 3:
                motifs.append(nom)
        if pire:
            res.append({"code": code, "nom": g["libelle_station"].iloc[0], "classe": pire,
                        "motif": ", ".join(motifs) if pire >= 3 else "", "annee": int(fin.year),
                        "lon": g["longitude"].median(), "lat": g["latitude"].median()})
    return pd.DataFrame(res)


def ipr_stations():
    an = dt.date.today().year
    champs = ["code_station", "date_operation", "ipr_code_classe", "coordonnee_x_station", "coordonnee_y_station",
              "code_epsg_projection_station"]
    morceaux = []
    for d in DEPARTEMENTS:
        r = get(f"{API}/v1/etat_piscicole/indicateurs", {
            "code_departement": d, "fields": ",".join(champs), "size": 20000,
            "date_operation_min": f"{an - 10}-01-01"}, timeout=300).json()
        if r.get("data"):
            morceaux.append(pd.DataFrame(r["data"]))
    if not morceaux:
        return pd.DataFrame(columns=["code", "classe", "annee", "x", "y"])
    df = pd.concat(morceaux, ignore_index=True)
    df["classe"] = pd.to_numeric(df["ipr_code_classe"], errors="coerce")
    df["date"] = pd.to_datetime(df["date_operation"], errors="coerce")
    df = df.dropna(subset=["classe", "date", "coordonnee_x_station", "coordonnee_y_station"])
    df = df[df["code_epsg_projection_station"].astype(str).isin(["2154", "2154.0"])]
    res = []
    for code, g in df.sort_values("date").groupby("code_station"):
        der = g.tail(3)           # les trois derniers indices : on garde la médiane
        res.append({"code": code, "classe": int(np.median(der["classe"]).round()), "annee": int(der["date"].iloc[-1].year),
                    "x": float(g["coordonnee_x_station"].median()), "y": float(g["coordonnee_y_station"].median())})
    return pd.DataFrame(res)


def main():
    eaux = gpd.read_file(DATA / "eaux.gpkg")
    out = pd.DataFrame({"id": eaux["id"], "q_classe": 0, "q_motif": "", "q_annee": 0, "q_station": "",
                        "ipr_classe": 0, "ipr_annee": 0})

    print("Qualité de l'eau (Hub'Eau, analyses physico-chimiques)...")
    try:
        st = qualite_stations(analyses())
    except Exception as e:      # complément : on continue sans
        print(f"  indisponible : {e}")
        st = pd.DataFrame()
    if len(st):
        print(f"  {len(st)} stations classées : {st['classe'].value_counts().sort_index().to_dict()}")
        pts = gpd.GeoSeries(gpd.points_from_xy(st["lon"], st["lat"]), crs=4326).to_crs(eaux.crs)
        k = stations_vers_rivieres(eaux, pts.x, pts.y)
        sel = k >= 0
        out.loc[sel, "q_classe"] = st["classe"].to_numpy()[k[sel]]
        out.loc[sel, "q_motif"] = st["motif"].to_numpy()[k[sel]]
        out.loc[sel, "q_annee"] = st["annee"].to_numpy()[k[sel]]
        out.loc[sel, "q_station"] = st["nom"].to_numpy()[k[sel]]
        print(f"  {int(sel.sum())} morceaux de rivière avec une qualité mesurée")

    print("Indice poissons rivière (OFB)...")
    try:
        ip = ipr_stations()
    except Exception as e:
        print(f"  indisponible : {e}")
        ip = pd.DataFrame()
    if len(ip):
        print(f"  {len(ip)} stations : {ip['classe'].value_counts().sort_index().to_dict()}")
        k = stations_vers_rivieres(eaux, ip["x"], ip["y"], meme_cours=3000)
        sel = k >= 0
        out.loc[sel, "ipr_classe"] = ip["classe"].to_numpy()[k[sel]]
        out.loc[sel, "ipr_annee"] = ip["annee"].to_numpy()[k[sel]]
        print(f"  {int(sel.sum())} morceaux de rivière avec un indice poissons")

    out = out[(out["q_classe"] > 0) | (out["ipr_classe"] > 0)]
    out.to_csv(DATA / "qualite.csv", index=False)


if __name__ == "__main__":
    main()
