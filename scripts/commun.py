"""Réglages communs : zone, espèces, téléchargements, petites fonctions de calcul."""
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import requests

RACINE = Path(__file__).resolve().parent.parent
DATA = RACINE / "data"
SITE = RACINE / "site"
DONNEES_SITE = SITE / "donnees"

# Bas-Rhin, Haut-Rhin, Meurthe-et-Moselle, Meuse, Moselle, Vosges
DEPARTEMENTS = ["67", "68", "54", "55", "57", "88"]
BBOX = (4.80, 47.40, 8.25, 49.65)        # lon min, lat min, lon max, lat max
CRS = "EPSG:2154"

# Grille d'altitude (100 m, Lambert-93), pour la pente des rivières et l'altitude des plans d'eau
X0, X1, Y0, Y1 = 830_000, 1_095_000, 6_700_000, 6_965_000
PAS = 100

# Espèces et codes des pêches scientifiques (Hub'Eau, OFB)
ESPECES = {
    "truite":  {"nom": "Truite fario", "codes": ["TRF"]},
    "ombre":   {"nom": "Ombre commun", "codes": ["OBR"]},
    "brochet": {"nom": "Brochet", "codes": ["BRO"]},
    "sandre":  {"nom": "Sandre", "codes": ["SAN"]},
    "perche":  {"nom": "Perche", "codes": ["PER"]},
    "carpe":   {"nom": "Carpe", "codes": ["CCO"]},
    "blancs":  {"nom": "Poissons blancs", "codes": ["GAR", "BRE", "CHE", "BAF", "TAN", "ROT"]},
    "silure":  {"nom": "Silure", "codes": ["SIL"]},
}

# Zone surveillée pour les prévisions
ZONE = {"nom": "autour de Marlenheim", "lat": 48.621, "lon": 7.491, "rayon_km": 40}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "poiscaille/1.0"


def get(url, params=None, essais=6, timeout=180, methode="GET", data=None):
    """Requête avec nouvelles tentatives en cas d'erreur réseau ou de limite."""
    attente = 5
    for i in range(essais):
        try:
            if methode == "POST":
                r = SESSION.post(url, data=data, timeout=timeout)
            else:
                r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code >= 400 and r.status_code != 206:
                r.raise_for_status()
            return r
        except (requests.RequestException, requests.HTTPError) as e:
            if i == essais - 1:
                raise
            print(f"  nouvelle tentative dans {attente} s ({e})")
            time.sleep(attente)
            attente = min(attente * 2, 120)


def sans_nan(o):
    """Valeurs manquantes (NaN, infini) remplacées par null : le JSON reste lisible par les navigateurs."""
    if isinstance(o, dict):
        return {k: sans_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sans_nan(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def ecrire_json(chemin, obj, compact=False):
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    obj = sans_nan(obj)
    texte = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) if compact else \
        json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False)
    Path(chemin).write_text(texte, encoding="utf-8")


def pic(v, lo, hi, marge, bord=0.75):
    """1 au centre de [lo, hi], « bord » aux extrémités, puis décroît jusqu'à 0 à « marge » en dehors."""
    v = np.asarray(v, dtype=float)
    c, demi = (lo + hi) / 2, max((hi - lo) / 2, 1e-6)
    dedans = 1 - (1 - bord) * ((v - c) / demi) ** 2
    d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
    dehors = bord * np.clip(1 - d / marge, 0, 1)
    return np.where((v >= lo) & (v <= hi), dedans, dehors)


def rampe(v, debut, fin):
    """0 avant « debut », 1 après « fin », linéaire entre les deux (fonctionne aussi à l'envers)."""
    v = np.asarray(v, dtype=float)
    return np.clip((v - debut) / (fin - debut), 0, 1)


# ---------- Réglementation nationale (à vérifier dans l'arrêté préfectoral et le règlement de l'AAPPMA) ----------

def _nieme_jour(annee, mois, jour_semaine, n):
    """n-ième jour de la semaine (0 = lundi) d'un mois ; n = -1 pour le dernier."""
    if n > 0:
        d = dt.date(annee, mois, 1)
        d += dt.timedelta(days=(jour_semaine - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    d = (dt.date(annee, mois + 1, 1) if mois < 12 else dt.date(annee + 1, 1, 1)) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - jour_semaine) % 7)


def periodes(annee):
    """Périodes d'ouverture nationales pour l'année : {espèce: {"1re": [(début, fin)], "2e": [(début, fin)]}}."""
    ouv_1re = _nieme_jour(annee, 3, 5, 2)            # 2e samedi de mars
    ferm_1re = _nieme_jour(annee, 9, 6, 3)           # 3e dimanche de septembre
    ombre = _nieme_jour(annee, 5, 5, 3)              # 3e samedi de mai
    dernier_dim_janv = _nieme_jour(annee, 1, 6, -1)
    dernier_sam_avril = _nieme_jour(annee, 4, 5, -1)
    an_debut, an_fin = dt.date(annee, 1, 1), dt.date(annee, 12, 31)
    carnassier_2e = [(an_debut, dernier_dim_janv), (dernier_sam_avril, an_fin)]
    general_1re = [(ouv_1re, ferm_1re)]
    return {
        "truite":  {"1re": general_1re, "2e": general_1re},
        "ombre":   {"1re": [(ombre, ferm_1re)], "2e": [(ombre, an_fin)]},
        "brochet": {"1re": [(dernier_sam_avril, ferm_1re)], "2e": carnassier_2e},
        "sandre":  {"1re": [(dernier_sam_avril, ferm_1re)], "2e": carnassier_2e},
        "perche":  {"1re": general_1re, "2e": [(an_debut, an_fin)]},
        "carpe":   {"1re": general_1re, "2e": [(an_debut, an_fin)]},
        "blancs":  {"1re": general_1re, "2e": [(an_debut, an_fin)]},
        "silure":  {"1re": general_1re, "2e": [(an_debut, an_fin)]},
    }


def ouvert(espece, jour):
    """(ouvert en 1re catégorie, ouvert en 2e catégorie, prochaine date d'ouverture si fermé partout)."""
    p = periodes(jour.year)[espece]
    o1 = any(a <= jour <= b for a, b in p["1re"])
    o2 = any(a <= jour <= b for a, b in p["2e"])
    prochain = None
    if not (o1 or o2):
        debuts = [a for cat in ("1re", "2e") for a, _ in p[cat] if a > jour]
        if not debuts:
            p2 = periodes(jour.year + 1)[espece]
            debuts = [a for cat in ("1re", "2e") for a, _ in p2[cat]]
        prochain = min(debuts)
    return o1, o2, prochain


def stations_vers_rivieres(eaux, sx, sy, meme_cours=5000, proche=300):
    """Pour chaque morceau de rivière ou canal : position de la station de mesure qui le représente, sinon -1.

    Une station est d'abord rattachée au morceau le plus proche (à « proche » m au plus) ; elle vaut ensuite pour
    les morceaux du même cours d'eau situés à moins de « meme_cours » m. En cas de choix, la plus proche l'emporte.
    """
    import geopandas as gpd
    import pandas as pd
    sx, sy = np.asarray(sx, float), np.asarray(sy, float)
    res = np.full(len(eaux), -1)
    est_riv = eaux["type"].isin(["riviere", "canal", "rhin"]).to_numpy()
    if len(sx) == 0 or not est_riv.any():
        return res
    e = eaux.reset_index(drop=True)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sx, sy), crs=e.crs)
    j = gpd.sjoin_nearest(pts, e.loc[est_riv, ["geometry"]], how="left", max_distance=proche, distance_col="dist")
    j = j[~j.index.duplicated()]
    piece = j["index_right"].reindex(range(len(sx)))
    noms = e["nom"].fillna("").astype(str).to_numpy()
    spiece = np.array([int(p) if pd.notna(p) else -1 for p in piece])
    snom = np.array([noms[p] if p >= 0 else "" for p in spiece])
    c = e.geometry.representative_point()
    cx, cy = c.x.to_numpy(), c.y.to_numpy()
    for i in np.where(est_riv)[0]:
        d = np.hypot(sx - cx[i], sy - cy[i])
        ok = (spiece == i) | ((snom == noms[i]) & (noms[i] != "") & (d <= meme_cours))
        if ok.any():
            res[i] = int(np.argmin(np.where(ok, d, np.inf)))
    return res
