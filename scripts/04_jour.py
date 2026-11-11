"""Étape quotidienne : conditions du jour et prévisions à 7 jours, pour chaque eau et chaque espèce.

Sources gratuites, sans clé :
- météo : Open-Meteo (30 jours passés + 7 jours de prévision) ;
- débits en temps réel : Hub'Eau hydrométrie (stations Vigicrues et DREAL) ;
- petits cours d'eau à sec ou presque : Hub'Eau écoulement (réseau ONDE de l'OFB, de mai à septembre).

Indice d'activité (0 à 1) = température de l'eau estimée × état du débit × pression × lune × réglementation.
La pression (baisse lente avant une perturbation, forte hausse après un front) et la lune (nouvelle et pleine lune)
ont un effet discuté : leur poids est volontairement modeste (quelques %).
La réglementation tient compte de la catégorie de l'eau quand elle est connue (étape 5).
Score du jour (0 à 1000) = potentiel × indice.

Résultats : site/donnees/jour_<espece>.json et site/donnees/info.json
"""
import datetime as dt
import json
import unicodedata
from zoneinfo import ZoneInfo

import ephem
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from commun import (BBOX, DATA, DEPARTEMENTS, DONNEES_SITE, ESPECES, ZONE, ecrire_json, get, ouvert, pic, rampe,
                    stations_vers_rivieres)

METEO = "https://api.open-meteo.com/v1/forecast"
HYDRO = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
ONDE = "https://hubeau.eaufrance.fr/api/v1/ecoulement/observations"
# Réseau ONDE : écoulement observé -> facteur (1 et 1a : normal ; 1f : faible ; 2 : sans courant visible ; 3 : à sec)
F_ONDE = {"1": 1.0, "1a": 1.0, "1f": 0.75, "2": 0.3, "3": 0.0}
PAS_METEO = (0.15, 0.12)
JOURS_PASSES, JOURS_PREVUS = 30, 7
GRADIENT = 0.0065
ESP = list(ESPECES)

# Température de l'eau idéale pour l'activité de chaque espèce (°C) : début, fin, marge
TEMP_EAU = {
    "truite": (8, 15, 5), "ombre": (8, 15, 5), "brochet": (8, 18, 7), "sandre": (12, 22, 6),
    "perche": (12, 22, 7), "carpe": (17, 26, 10), "blancs": (12, 22, 8), "silure": (18, 27, 10),
}
# Espèces qui profitent d'une légère montée des eaux (eau teintée), selon l'expérience des pêcheurs
AIMENT_MONTEE = {"truite", "brochet", "perche", "silure", "blancs"}
# Variation de pression sur 24 h (hPa) -> facteur : baisse lente = un peu mieux, forte hausse ou chute brutale = moins bien
PRESSION_DP = [-12, -8, -4, -1.5, 0, 2, 5, 10]
PRESSION_F = [0.88, 0.95, 1.08, 1.06, 1.0, 0.98, 0.9, 0.86]
SENS_PRESSION = {"truite": 1.0, "ombre": 0.6, "brochet": 1.0, "sandre": 0.8, "perche": 1.0, "carpe": 0.7,
                 "blancs": 0.6, "silure": 0.8}
# Lune : jusqu'à ±5 % (nouvelle et pleine lune un peu meilleures que les quartiers, théorie solunaire)
SENS_LUNE = {"truite": 0.6, "ombre": 0.5, "brochet": 0.6, "sandre": 0.8, "perche": 0.6, "carpe": 1.0,
             "blancs": 0.5, "silure": 1.0}
STATUTS = {"public": "p", "aappma": "a", "prive": "x"}       # autre : « i » (non renseigné)
ORDRE_STATUTS = "paxi"
# Température de l'eau : moyenne de l'air sur n jours, pente, décalage (approximation)
INERTIE = {"riviere": (5, 0.75, 2.5), "canal": (10, 0.8, 2.5), "rhin": (10, 0.8, 3.0), "etang": (10, 0.85, 2.0),
           "port": (10, 0.85, 2.5), "lac": (20, 0.85, 2.0), "graviere": (20, 0.85, 2.0), "retenue": (20, 0.85, 2.0)}


def sans_accents(t):
    t = unicodedata.normalize("NFD", str(t or "")).encode("ascii", "ignore").decode().lower()
    for article in ["la ", "le ", "l'", "les "]:
        if t.startswith(article):
            t = t[len(article):]
    return t.strip()


def points_meteo():
    lons = np.arange(BBOX[0], BBOX[2] + 1e-9, PAS_METEO[0])
    lats = np.arange(BBOX[1], BBOX[3] + 1e-9, PAS_METEO[1])       # du sud au nord
    return lons, lats


def meteo():
    lons, lats = points_meteo()
    pts = [(lo, la) for la in lats for lo in lons]
    variables = ["temperature_2m_mean", "precipitation_sum", "pressure_msl_mean", "wind_speed_10m_max"]
    series = {v: [] for v in variables}
    alt = []
    for i in range(0, len(pts), 100):
        lot = pts[i:i + 100]
        data = get(METEO, {
            "longitude": ",".join(f"{p[0]:.3f}" for p in lot), "latitude": ",".join(f"{p[1]:.3f}" for p in lot),
            "daily": ",".join(variables), "past_days": JOURS_PASSES, "forecast_days": JOURS_PREVUS + 1,
            "timezone": "Europe/Paris"}).json()
        data = data if isinstance(data, list) else [data]
        for d in data:
            for v in variables:
                series[v].append(np.array(d["daily"].get(v, [np.nan] * (JOURS_PASSES + JOURS_PREVUS + 1)), dtype=float))
            alt.append(float(d.get("elevation") or 0))
    forme = (len(lats), len(lons))
    out = {v: np.array(series[v]).reshape(forme + (-1,)) for v in variables}
    out["alt"] = np.array(alt).reshape(forme)
    return out, lons, lats


def interpoler(champ, lons, lats, lon, lat):
    champ = np.where(np.isnan(champ), np.nanmean(champ), champ)
    f = RegularGridInterpolator((lats, lons), champ, bounds_error=False, fill_value=None)
    return f(np.column_stack([lat, lon]))


def une_station(s, maintenant):
    """Débit actuel, il y a 24 h, et débit habituel à cette période (médiane des 3 dernières années)."""
    try:
        tr = get(f"{HYDRO}/observations_tr", {"code_entite": s["code_station"], "grandeur_hydro": "Q", "size": 400,
                                              "sort": "desc", "fields": "date_obs,resultat_obs"},
                 essais=2, timeout=60).json()["data"]
    except Exception:
        return None
    obs = pd.DataFrame(tr)
    if obs.empty:
        return None
    obs["date_obs"] = pd.to_datetime(obs["date_obs"], utc=True)
    obs = obs.dropna(subset=["resultat_obs"])
    if obs.empty or (maintenant - obs.date_obs.iloc[0]).total_seconds() > 2 * 86400:
        return None
    q = float(obs.resultat_obs.iloc[0])
    avant = obs[obs.date_obs <= obs.date_obs.iloc[0] - pd.Timedelta(hours=24)]
    q24 = float(avant.resultat_obs.iloc[0]) if len(avant) else np.nan
    try:
        el = get(f"{HYDRO}/obs_elab", {"code_entite": s["code_station"], "grandeur_hydro_elab": "QmnJ", "size": 2000,
                                       "date_debut_obs_elab": (maintenant - dt.timedelta(days=3 * 365)).date().isoformat(),
                                       "fields": "date_obs_elab,resultat_obs_elab"}, essais=2, timeout=60).json()["data"]
    except Exception:
        el = []
    e = pd.DataFrame(el).dropna() if el else pd.DataFrame()
    if e.empty:
        return None
    jour = pd.to_datetime(e["date_obs_elab"]).dt.dayofyear
    ecart = np.abs(((jour - maintenant.timetuple().tm_yday + 182) % 365) - 182)
    proche = e[ecart <= 20]["resultat_obs_elab"]
    norm = float(proche.median()) if len(proche) >= 10 else float(e["resultat_obs_elab"].median())
    if not norm or norm <= 0:
        return None
    return {"code": s["code_station"], "nom": s["libelle_station"], "riviere": s.get("libelle_cours_eau") or "",
            "lon": s["longitude_station"], "lat": s["latitude_station"],
            "ratio": q / norm, "tendance": (q / q24) if q24 and q24 > 0 else 1.0}


def stations_hydro():
    from concurrent.futures import ThreadPoolExecutor
    maintenant = dt.datetime.now(dt.timezone.utc)
    liste = []
    for d in DEPARTEMENTS:
        liste += get(f"{HYDRO}/referentiel/stations", {
            "code_departement": d, "en_service": "true", "size": 1000,
            "fields": "code_station,libelle_station,libelle_cours_eau,longitude_station,latitude_station"}).json()["data"]
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda s: une_station(s, maintenant), liste))
    return pd.DataFrame([r for r in res if r])


def etat_debit(ratio):
    if ratio > 3:
        return "crue"
    if ratio > 1.8:
        return "hautes eaux"
    if ratio >= 0.6:
        return "normal"
    if ratio >= 0.35:
        return "basses eaux"
    return "étiage sévère"


def rattacher(eaux, st):
    """Station de débit de chaque morceau de rivière : même cours d'eau à 40 km, sinon la plus proche à 15 km."""
    n = len(eaux)
    ratio, tend = np.full(n, np.nan), np.full(n, 1.0)
    if st.empty:
        return ratio, tend
    t = Transformer.from_crs(4326, 2154, always_xy=True)
    sx, sy = t.transform(st.lon.to_numpy(), st.lat.to_numpy())
    snom = st.riviere.map(sans_accents).to_numpy()
    c = eaux.geometry.representative_point()
    for i, (x, y, nom, typ) in enumerate(zip(c.x, c.y, eaux["nom"].map(sans_accents), eaux["type"])):
        if typ not in ("riviere", "canal", "rhin"):
            continue
        d = np.hypot(sx - x, sy - y)
        meme = (snom == nom) & (nom != "") & (d < 40_000)
        j = np.argmin(np.where(meme, d, np.inf)) if meme.any() else (np.argmin(d) if d.min() < 15_000 else -1)
        if j >= 0:
            ratio[i], tend[i] = st.ratio.iloc[j], st.tendance.iloc[j]
    return ratio, tend


PHASES = [("Nouvelle lune", "🌑"), ("Premier croissant", "🌒"), ("Premier quartier", "🌓"),
          ("Gibbeuse croissante", "🌔"), ("Pleine lune", "🌕"), ("Gibbeuse décroissante", "🌖"),
          ("Dernier quartier", "🌗"), ("Dernier croissant", "🌘")]


def lune(jour):
    paris = ZoneInfo("Europe/Paris")
    d = ephem.Date(dt.datetime(jour.year, jour.month, jour.day, 12, tzinfo=paris).astimezone(dt.timezone.utc).replace(tzinfo=None))
    avant, apres = ephem.previous_new_moon(d), ephem.next_new_moon(d)
    nom, emoji = PHASES[int(round((d - avant) / (apres - avant) * 8)) % 8]
    loc = lambda e: e.datetime().replace(tzinfo=dt.timezone.utc).astimezone(paris).date().isoformat()
    return {"phase": nom, "emoji": emoji, "illumination": round(ephem.Moon(d).phase),
            "croissante": ephem.next_full_moon(d) < apres, "prochaine_nouvelle": loc(apres),
            "prochaine_pleine": loc(ephem.next_full_moon(d))}


def age_lune(jour):
    """Âge de la lune à midi, en fraction du cycle (0 : nouvelle lune, 0,5 : pleine lune)."""
    d = ephem.Date(dt.datetime(jour.year, jour.month, jour.day, 11))
    avant, apres = ephem.previous_new_moon(d), ephem.next_new_moon(d)
    return float((d - avant) / (apres - avant))


def facteur_pression(dp24, sens):
    f = np.interp(np.nan_to_num(np.asarray(dp24, float), nan=0.0), PRESSION_DP, PRESSION_F)
    return 1 + sens * (f - 1)


def facteur_lune(age, sens):
    return 1 + sens * 0.05 * np.cos(4 * np.pi * age)


def sous_ensembles():
    """Toutes les combinaisons de statuts (« p », « pa », « paxi »...) pour les filtres du site."""
    return ["".join(c for j, c in enumerate(ORDRE_STATUTS) if m >> j & 1) for m in range(1, 2 ** len(ORDRE_STATUTS))]


def lire_acces(eaux):
    chemin = DATA / "acces.csv"
    st = pd.Series("i", index=eaux.index)
    cat = pd.Series(0, index=eaux.index)
    if chemin.exists():
        a = pd.read_csv(chemin, dtype={"dep": str}, keep_default_na=False).drop_duplicates("id").set_index("id")
        st = eaux["id"].map(a["statut"].map(STATUTS)).fillna("i")
        cat = pd.to_numeric(eaux["id"].map(a["cat"]), errors="coerce").fillna(0).astype(int)
    return st.to_numpy(), cat.to_numpy()


def ecoulements(eaux, jour):
    """Dernière observation du réseau ONDE (30 derniers jours) pour les petits cours d'eau : facteur et détail."""
    debut = (jour - dt.timedelta(days=JOURS_PASSES)).isoformat()
    lignes = []
    for d in DEPARTEMENTS:
        try:
            r = get(ONDE, {"code_departement": d, "date_observation_min": debut, "size": 5000,
                           "fields": "code_station,libelle_station,libelle_cours_eau,date_observation,code_ecoulement,"
                                     "libelle_ecoulement,longitude,latitude"}, timeout=120).json()
            lignes += r.get("data") or []
        except Exception as e:      # complément : on continue sans
            print(f"  écoulements indisponibles ({d}) : {e}")
    f, detail = np.ones(len(eaux)), {}
    if not lignes:
        return f, detail, []
    df = pd.DataFrame(lignes).dropna(subset=["longitude", "latitude", "code_ecoulement"])
    df = df.sort_values("date_observation").groupby("code_station").tail(1).reset_index(drop=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(df["longitude"], df["latitude"]), crs=4326).to_crs(eaux.crs)
    k = stations_vers_rivieres(eaux, pts.x, pts.y, meme_cours=2000, proche=200)
    for i in np.where(k >= 0)[0]:
        code = str(df["code_ecoulement"].iloc[k[i]])
        f[i] = F_ONDE.get(code, 1.0)
        if f[i] < 1:
            detail[str(int(eaux["id"].iloc[i]))] = [code, str(df["date_observation"].iloc[k[i]])[:10]]
    texte = lambda v: "" if pd.isna(v) else str(v)
    obs = [{"nom": texte(r.libelle_station), "riviere": texte(r.libelle_cours_eau), "lat": round(float(r.latitude), 5),
            "lon": round(float(r.longitude), 5), "code": str(r.code_ecoulement), "libelle": texte(r.libelle_ecoulement),
            "date": str(r.date_observation)[:10]} for r in df.itertuples()]
    print(f"  {len(df)} stations ONDE observées depuis le {debut} ; {len(detail)} morceaux à sec ou presque")
    return f, detail, obs


def texte_reglementation(espece, jour):
    o1, o2, prochain = ouvert(espece, jour)
    if o1 and o2:
        return "ouverte en 1re et 2e catégorie", True
    if o2:
        return "ouverte en 2e catégorie, fermée en 1re catégorie", True
    if o1:
        return "ouverte en 1re catégorie, fermée en 2e catégorie", True
    return f"fermée jusqu'au {prochain.strftime('%d/%m/%Y')}", False


def main():
    eaux = gpd.read_file(DATA / "eaux.gpkg")
    pot = pd.read_csv(DATA / "potentiel.csv").set_index("id").loc[eaux["id"]]
    aujourdhui = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    dates = [aujourdhui + dt.timedelta(days=k) for k in range(JOURS_PREVUS + 1)]
    print(f"Conditions du {aujourdhui} et prévisions à {JOURS_PREVUS} jours...")

    m, lons, lats = meteo()
    c = eaux.geometry.representative_point().to_crs(4326)
    lon, lat = c.x.to_numpy(), c.y.to_numpy()
    alt_grille = interpoler(m["alt"], lons, lats, lon, lat)
    alt_eaux = eaux["altitude"].to_numpy(float)
    ecart = alt_grille - np.where(np.isnan(alt_eaux), alt_grille, alt_eaux)
    tair = interpoler(m["temperature_2m_mean"], lons, lats, lon, lat) + GRADIENT * ecart[:, None]
    pluie = np.nan_to_num(interpoler(m["precipitation_sum"], lons, lats, lon, lat))

    print("Débits (Hub'Eau)...")
    st = stations_hydro()
    print(f"  {len(st)} stations avec un débit récent")
    ratio, tend = rattacher(eaux, st)
    est_riv = eaux["type"].isin(["riviere", "canal", "rhin"]).to_numpy()
    f_deb = np.ones(len(eaux))
    connu = est_riv & ~np.isnan(ratio)
    f_deb[connu] = np.where(ratio[connu] > 1.8, 1 - 0.7 * rampe(ratio[connu], 1.8, 4), 1.0)
    f_deb[connu] = np.where(ratio[connu] < 0.5, 0.75, f_deb[connu])
    montee = connu & (tend >= 1.1) & (tend <= 1.8) & (np.nan_to_num(ratio) < 2)
    print("Écoulement des petits cours d'eau (ONDE)...")
    f_onde, assecs, obs_onde = ecoulements(eaux, aujourdhui)
    # Une observation de cours d'eau à sec ne vaut plus après de bonnes pluies (plus de 25 mm depuis)
    ids = eaux["id"].astype(int).astype(str).to_numpy()
    for i in np.where(f_onde < 1)[0]:
        n = (aujourdhui - dt.date.fromisoformat(assecs[ids[i]][1])).days
        if n > 0 and pluie[i, max(0, JOURS_PASSES - n + 1):JOURS_PASSES + 1].sum() > 25:
            f_onde[i] = 1.0
            assecs.pop(ids[i], None)
    print(f"  {len(assecs)} morceaux encore à sec ou presque après les pluies récentes")

    # Température de l'eau estimée, pour chaque eau et chaque jour (aujourd'hui = indice JOURS_PASSES)
    teau = np.zeros((len(eaux), JOURS_PREVUS + 1))
    for typ, (n, a, b) in INERTIE.items():
        sel = (eaux["type"] == typ).to_numpy()
        if not sel.any():
            continue
        for k in range(JOURS_PREVUS + 1):
            i = JOURS_PASSES + k
            teau[sel, k] = a * np.nanmean(tair[sel, i - n + 1:i + 1], axis=1) + b
    teau = np.clip(teau, 0.5, 28)

    pot_zone = Transformer.from_crs(4326, 2154, always_xy=True).transform(ZONE["lon"], ZONE["lat"])
    cl = eaux.geometry.representative_point()
    dans_zone = np.hypot(cl.x - pot_zone[0], cl.y - pot_zone[1]).to_numpy() <= ZONE["rayon_km"] * 1000

    # Pression : variation sur 24 h de la moyenne journalière ; lune : âge du cycle
    pression = interpoler(m["pressure_msl_mean"], lons, lats, lon, lat)
    dp24 = np.stack([pression[:, JOURS_PASSES + k] - pression[:, JOURS_PASSES + k - 1] for k in range(len(dates))], axis=1)
    ages = [age_lune(j) for j in dates]
    statut_eau, cat_eau = lire_acces(eaux)
    filtres = {cle: np.isin(statut_eau, list(cle)) for cle in sous_ensembles()}
    print("Statuts :", {c: int((statut_eau == c).sum()) for c in ORDRE_STATUTS})

    DONNEES_SITE.mkdir(parents=True, exist_ok=True)
    previsions, reglementation, statuts = {}, {}, {}
    for e in ESP:
        p = pot[e].to_numpy(float)
        jours, prev = [], {"zone": [], "region": [], "statut": [], "filtres": {c: [] for c in filtres},
                           "f_pression": [], "f_lune": []}
        for k, jour in enumerate(dates):
            texte, legal = texte_reglementation(e, jour)
            o1, o2, _ = ouvert(e, jour)
            # Catégorie connue : sa propre période d'ouverture ; sinon ouverte si l'une des deux l'est
            permis = np.where(cat_eau == 1, o1, np.where(cat_eau == 2, o2, o1 or o2)).astype(float)
            f_t = pic(teau[:, k], *TEMP_EAU[e])
            f_p = facteur_pression(dp24[:, k], SENS_PRESSION[e])
            f_l = float(facteur_lune(ages[k], SENS_LUNE[e]))
            # Débit : connu aujourd'hui, revient peu à peu à la normale dans les prévisions ; grosse pluie prévue = risque de crue
            f_d = 1 - (1 - f_deb) * max(0.0, 1 - k / 4)
            i = JOURS_PASSES + k
            pluie_2j = pluie[:, i - 1:i + 1].sum(axis=1)
            f_d = np.where(est_riv & (pluie_2j > 25), f_d * 0.8, f_d)
            bonus = np.where(montee & (k <= 1) & (e in AIMENT_MONTEE), 1.1, 1.0)
            indice = np.clip(f_t * f_d * bonus * f_p * f_l, 0, 1) * permis * f_onde
            score = np.rint(p * indice).astype(int)
            jours.append(score.tolist())
            bons = p >= 400
            for nom_zone, sel in (("zone", bons & dans_zone), ("region", bons)):
                prev[nom_zone].append(int(np.percentile(score[sel], 90)) if sel.sum() >= 10 else 0)
            for cle, sel in filtres.items():
                sel = sel & bons & dans_zone
                n_sel = int(sel.sum())
                prev["filtres"][cle].append(None if n_sel == 0 else int(np.percentile(score[sel], 90 if n_sel >= 10 else 75)))
            z = dans_zone if dans_zone.any() else np.ones(len(eaux), bool)
            prev["f_pression"].append(round(float(np.nanmedian(f_p[z])), 3))
            prev["f_lune"].append(round(f_l, 3))
            v = float(np.percentile(indice[bons], 90)) if bons.sum() >= 10 else 0
            prev["statut"].append("pêche fermée" if not legal else
                                  "bon moment" if v >= 0.55 else "conditions moyennes" if v >= 0.3 else "peu favorable")
            if k == 0:
                reglementation[e] = texte
        ecrire_json(DONNEES_SITE / f"jour_{e}.json", {"j": jours}, compact=True)
        previsions[e] = prev
        statuts[e] = prev["statut"][0]
        print(f"{ESPECES[e]['nom']} : {prev['statut'][0]} ; {ZONE['nom']} {prev['zone']} ; réglementation : {reglementation[e]}")

    a = JOURS_PASSES
    with np.errstate(invalid="ignore"):
        dp = np.nanmedian(pression[dans_zone, a] - pression[dans_zone, a - 3]) if dans_zone.any() else np.nan
    riv_z, pl_z = dans_zone & est_riv, dans_zone & ~est_riv
    z_all = dans_zone if dans_zone.any() else np.ones(len(eaux), bool)
    ecrire_json(DONNEES_SITE / "info.json", {
        "date": aujourdhui.isoformat(),
        "dates": [d.isoformat() for d in dates],
        "zone": ZONE,
        "especes": {e: {"nom": ESPECES[e]["nom"], "statut": statuts[e], "reglementation": reglementation[e]} for e in ESP},
        "previsions": previsions,
        "meteo": {
            "air_7j": round(float(np.nanmean(tair[dans_zone, a - 6:a + 1])), 1),
            "eau_rivieres": round(float(np.median(teau[riv_z, 0])), 1) if riv_z.any() else None,
            "eau_plans": round(float(np.median(teau[pl_z, 0])), 1) if pl_z.any() else None,
            "pluie_7j_mm": round(float(np.mean(pluie[dans_zone, a - 7:a].sum(axis=1))), 1),
            "pluie_prevue_7j_mm": round(float(np.mean(pluie[dans_zone, a + 1:].sum(axis=1))), 1),
            "pression_3j_hpa": None if np.isnan(dp) else round(float(dp), 1),
        },
        "stations": [{"nom": s.nom, "riviere": s.riviere, "lat": round(s.lat, 5), "lon": round(s.lon, 5),
                      "ratio": round(float(s.ratio), 2), "tendance": round(float(s.tendance), 2), "etat": etat_debit(s.ratio)}
                     for s in st.itertuples()],
        "lune": lune(aujourdhui),
        "fiabilite": json.loads((DATA / "fiabilite.json").read_text(encoding="utf-8")) if (DATA / "fiabilite.json").exists() else {},
        "assecs": assecs,
        "onde": obs_onde,
        "jours": [{"pression_hpa": round(float(np.nanmedian(pression[z_all, JOURS_PASSES + k])), 1),
                   "dp24_hpa": round(float(np.nanmedian(dp24[z_all, k])), 1),
                   "lune": lune(d)["phase"], "emoji": lune(d)["emoji"], "illumination": lune(d)["illumination"]}
                  for k, d in enumerate(dates)],
    })
    print("Conditions du jour et prévisions prêtes.")


if __name__ == "__main__":
    main()
