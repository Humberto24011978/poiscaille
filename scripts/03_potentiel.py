"""Étape 3 : potentiel de chaque rivière et plan d'eau pour chaque espèce (score de 0 à 1000).

- Rivières et canaux : un modèle (forêt aléatoire) apprend sur les pêches scientifiques où vit chaque espèce
  (altitude, pente, taille du cours d'eau, position), combiné à des règles de zonation piscicole
  (zone à truite en tête de bassin, zone à ombre, à barbeau, puis à brème en plaine).
- Plans d'eau : règles selon le type (étang, gravière, lac de montagne, grande retenue), la surface et l'altitude.
- Un lieu de pêche signalé (OpenStreetMap), surtout d'une association, donne un petit bonus.

Résultats :
- data/potentiel.csv : scores par objet et par espèce
- site/donnees/eaux.json : la carte des eaux pour le site (géométries simplifiées et scores)
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from sklearn.ensemble import RandomForestClassifier

from commun import DATA, DONNEES_SITE, ESPECES, ecrire_json, pic, rampe

ESP = list(ESPECES)
MIN_PRESENCES = 15


def regles_rivieres(e):
    """Aptitude (0 à 1) de chaque morceau de rivière ou canal, d'après la zonation piscicole."""
    pente = e["pente_pm"].to_numpy(float)            # pour mille
    ordre = e["ordre"].to_numpy(float)
    larg = e["largeur"].to_numpy(float)              # 1 : < 5 m ; 2 : 5-15 m ; 3 : 15-50 m ; 4 : > 50 m
    alt = e["altitude"].to_numpy(float)
    canal = (e["type"] == "canal").to_numpy()
    rhin = (e["type"] == "rhin").to_numpy()
    grand = np.maximum(rampe(ordre, 4, 7), rampe(larg, 2, 4))
    lent = 1 - rampe(pente, 0.5, 3)                  # 1 sous 0,5 ‰, 0 au-dessus de 3 ‰
    vif = rampe(pente, 1, 6)
    r = {
        "truite":  vif * (1 - rampe(ordre, 5, 7)) * (0.4 + 0.6 * rampe(alt, 150, 400)),
        "ombre":   pic(pente, 1.5, 6, 3) * pic(ordre, 3.5, 5.5, 2) * pic(alt, 180, 550, 250),
        "brochet": lent * np.maximum(grand, 0.4),
        "sandre":  lent * grand,
        "perche":  (1 - rampe(pente, 1, 5)) * np.maximum(grand, 0.4),
        "carpe":   lent * grand,
        "blancs":  (1 - rampe(pente, 3, 10)) * np.maximum(rampe(ordre, 2, 5), 0.3),
        "silure":  lent * rampe(ordre, 5, 8) * rampe(larg, 2, 4),
    }
    for k in r:
        if k in ("truite", "ombre"):
            r[k] = np.where(canal, 0.05, r[k]) * np.where(rhin, 0.2, 1)
        else:
            r[k] = np.where(canal, np.maximum(r[k], 0.6 if k != "silure" else 0.4), r[k])
            r[k] = np.where(rhin, np.maximum(r[k], 0.85), r[k])
    return {k: np.clip(v, 0, 1) for k, v in r.items()}


# Aptitude des plans d'eau par type
PLANS = {
    #            truite ombre brochet sandre perche carpe blancs silure
    "etang":    [0.10, 0.0, 0.60, 0.25, 0.70, 0.95, 0.95, 0.15],
    "graviere": [0.10, 0.0, 0.85, 0.65, 0.90, 0.85, 0.80, 0.55],
    "lac":      [0.75, 0.05, 0.70, 0.25, 0.80, 0.35, 0.65, 0.05],
    "retenue":  [0.30, 0.0, 0.80, 0.75, 0.85, 0.80, 0.85, 0.40],
    "port":     [0.0, 0.0, 0.60, 0.80, 0.80, 0.85, 0.85, 0.80],
}


def regles_plans(e):
    base = np.array([PLANS.get(t, PLANS["etang"]) for t in e["type"]])
    s = e["surface_ha"].to_numpy(float)
    alt = e["altitude"].to_numpy(float)
    r = {k: base[:, i].copy() for i, k in enumerate(ESP)}
    taille = rampe(np.log10(np.maximum(s, 0.01)), -0.5, 1.5)      # 0,3 ha -> faible, 30 ha et plus -> 1
    r["sandre"] *= 0.4 + 0.6 * taille
    r["silure"] *= taille
    r["brochet"] *= 0.6 + 0.4 * taille
    montagne = rampe(alt, 400, 700)
    r["truite"] = np.maximum(r["truite"], 0.7 * montagne)            # lacs et étangs d'altitude
    r["carpe"] *= 1 - 0.6 * montagne
    r["sandre"] *= 1 - 0.7 * montagne
    r["silure"] *= 1 - 0.9 * montagne
    return {k: np.clip(v, 0, 1) for k, v in r.items()}


def variables(e):
    c = e.geometry.representative_point()
    return np.column_stack([
        e["altitude"], np.log1p(e["pente_pm"]), e["ordre"], e["largeur"],
        (e["type"] == "canal").astype(int), (e["type"] == "rhin").astype(int), c.x / 1000, c.y / 1000,
    ]).astype("float32")


def arrondir(c):
    if isinstance(c, (float, int)):
        return round(float(c), 5)
    return [arrondir(x) for x in c]


def main():
    eaux = gpd.read_file(DATA / "eaux.gpkg")
    peches = pd.read_csv(DATA / "peches.csv")
    est_riv = eaux["type"].isin(["riviere", "canal", "rhin"]).to_numpy()
    riv, plans = eaux[est_riv].copy(), eaux[~est_riv].copy()
    scores = {k: np.zeros(len(eaux)) for k in ESP}

    # Stations de pêche scientifique rattachées au morceau de rivière le plus proche (à 300 m au plus)
    st = gpd.GeoDataFrame(peches, geometry=gpd.points_from_xy(peches.x, peches.y), crs=eaux.crs)
    st = gpd.sjoin_nearest(st, riv[["geometry"]], how="inner", max_distance=300, distance_col="dist")
    st = st[~st.index.duplicated()]
    print(f"{len(st)} stations rattachées à une rivière (sur {len(peches)})")
    X_riv = variables(riv)
    pos = {idx: i for i, idx in enumerate(riv.index)}
    lignes = np.array([pos[i] for i in st["index_right"]])

    r_riv = regles_rivieres(riv)
    r_pl = regles_plans(plans) if len(plans) else {k: np.array([]) for k in ESP}
    resume = {}
    for k in ESP:
        y = (st[f"{k}_n"] > 0).to_numpy().astype(int)
        regle = r_riv[k]
        if y.sum() >= MIN_PRESENCES and (len(y) - y.sum()) >= MIN_PRESENCES:
            rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.6, n_jobs=-1,
                                        random_state=42)
            rf.fit(X_riv[lignes], y)
            proba = rf.predict_proba(X_riv)[:, 1]
            proba = np.clip(proba / max(np.percentile(proba, 98), 1e-6), 0, 1)
            final = 0.65 * proba + 0.35 * regle
            info = f"modèle appris sur {int(y.sum())} stations où il est présent et {int(len(y) - y.sum())} où il est absent"
        else:
            final = regle
            info = f"règles seules ({int(y.sum())} stations où il est présent)"
        scores[k][est_riv] = final
        if len(plans):
            scores[k][~est_riv] = r_pl[k]
        resume[k] = info
        print(f"{ESPECES[k]['nom']} : {info}")

    # Petit bonus pour les lieux de pêche signalés (davantage s'ils relèvent d'une association)
    bonus = 1 + 0.10 * eaux["peche"].to_numpy() + 0.10 * eaux["aappma"].to_numpy()
    out = pd.DataFrame({"id": eaux["id"]})
    for k in ESP:
        v = scores[k] * bonus
        ref = np.percentile(v[v > 0], 99.5) if (v > 0).any() else 1
        out[k] = np.rint(np.clip(v / max(ref, 1e-6), 0, 1) ** 1.2 * 1000).astype(int)
    out.to_csv(DATA / "potentiel.csv", index=False)

    # Espèces vues lors des pêches scientifiques, rattachées au morceau de rivière
    vues = {}
    for _, s in st.iterrows():
        liste = [f"{k}:{int(s[f'{k}_derniere'])}" for k in ESP if s[f"{k}_n"] > 0]
        vues.setdefault(int(eaux.loc[s["index_right"], "id"]), set()).update(liste)

    # Carte pour le site : géométries simplifiées, coordonnées en degrés
    affiche = eaux.copy()
    affiche["geometry"] = affiche.geometry.simplify(15)
    affiche = affiche.to_crs(4326)
    features = []
    for i, row in affiche.iterrows():
        g = mapping(row.geometry)
        g = {"type": g["type"], "coordinates": arrondir(g["coordinates"])}
        p = {"i": int(row["id"]), "t": row["type"], "a": int(row["altitude"] or 0),
             "p": [int(out.loc[i, k]) for k in ESP]}
        if row["nom"]:
            p["n"] = row["nom"]
        if row["type"] in ("riviere", "canal", "rhin"):
            p["o"], p["w"] = int(row["ordre"]), int(row["largeur"])
        else:
            p["s"] = float(row["surface_ha"])
        if row["peche"]:
            p["f"] = 2 if row["aappma"] else 1
        if int(row["id"]) in vues:
            p["v"] = sorted(vues[int(row["id"])])
        features.append({"type": "Feature", "properties": p, "geometry": g})
    ecrire_json(DONNEES_SITE / "eaux.json", {"type": "FeatureCollection", "especes": ESP, "features": features}, compact=True)
    ecrire_json(DATA / "potentiel_resume.json", resume)
    taille = (DONNEES_SITE / "eaux.json").stat().st_size / 1e6
    print(f"Carte des eaux : {len(features)} objets, {taille:.1f} Mo")


if __name__ == "__main__":
    main()
