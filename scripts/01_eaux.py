"""Étape 1 : rivières, canaux et plans d'eau de la zone, avec leurs caractéristiques.

Sources gratuites, sans clé :
- tronçons de cours d'eau et plans d'eau : BD TOPO de l'IGN (service WFS de la Géoplateforme) ;
- limites départementales : ADMIN EXPRESS (IGN) ;
- altitude : Copernicus GLO-30 (pour la pente des rivières et l'altitude des plans d'eau) ;
- lieux de pêche signalés : OpenStreetMap (leisure=fishing), qui mentionnent parfois l'AAPPMA.

Résultat : data/eaux.gpkg (un objet par morceau de rivière d'au plus 2,5 km et par plan d'eau)
"""
import os
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely import force_2d
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring

from commun import BBOX, CRS, DATA, DEPARTEMENTS, PAS, X0, X1, Y0, Y1, get

WFS = "https://data.geopf.fr/wfs/ows"
OVERPASS = "https://overpass-api.de/api/interpreter"
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
TUILE = 0.25
PAGE = 5000
LONGUEUR_MAX = 2500       # m : longueur maximale d'un morceau de rivière
SURFACE_MIN = 3000        # m² : plans d'eau plus petits ignorés (0,3 ha)

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")

NATURES_COURS = {"Ecoulement naturel", "Canal", "Ecoulement canalisé"}
NATURES_PLANS = {
    "Lac": "lac", "Retenue": "etang", "Retenue-barrage": "retenue", "Retenue-digue": "etang",
    "Plan d'eau de gravière": "graviere", "Réservoir-bassin piscicole": "etang",
    "Retenue-bassin portuaire": "port", "Plan d'eau de mine": "graviere",
}
LARGEUR = {"Entre 0 et 5 m": 1, "Entre 5 et 15 m": 2, "Entre 15 et 50 m": 3, "Plus de 50 m": 4}


def wfs(couche, bbox, tri=None, champs=None):
    morceaux, debut = [], 0
    while True:
        params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": couche,
                  "outputFormat": "application/json", "srsName": CRS, "count": PAGE, "startIndex": debut,
                  "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326"}
        if tri:
            params["sortBy"] = tri
        if champs:
            params["propertyName"] = ",".join(champs + ["geometrie"])
        gdf = gpd.read_file(BytesIO(get(WFS, params, timeout=300).content))
        if gdf.empty:
            break
        morceaux.append(gdf)
        if len(gdf) < PAGE:
            break
        debut += PAGE
    if not morceaux:
        return gpd.GeoDataFrame(geometry=[], crs=CRS)
    return gpd.GeoDataFrame(pd.concat(morceaux, ignore_index=True), geometry="geometry", crs=CRS)


def par_tuiles(couche, emprise, nom, cle="cleabs", champs=None):
    parties = []
    lon1, lat1, lon2, lat2 = emprise
    tuiles = [(round(lo, 4), round(la, 4), round(min(lo + TUILE, lon2), 4), round(min(la + TUILE, lat2), 4))
              for lo in np.arange(lon1, lon2, TUILE) for la in np.arange(lat1, lat2, TUILE)]
    for i, t in enumerate(tuiles, start=1):
        g = wfs(couche, t, tri=cle, champs=champs)
        if not g.empty:
            parties.append(g)
        if i % 20 == 0 or i == len(tuiles):
            print(f"  {nom} : tuile {i}/{len(tuiles)}")
    tout = pd.concat(parties, ignore_index=True).drop_duplicates(subset=cle)
    return gpd.GeoDataFrame(tout, geometry="geometry", crs=CRS)


def departements():
    g = wfs("ADMINEXPRESS-COG-CARTO.LATEST:departement", BBOX)
    g = g[g["code_insee"].astype(str).isin(DEPARTEMENTS)]
    if len(g) != len(DEPARTEMENTS):
        raise SystemExit(f"Départements trouvés : {sorted(g.code_insee)} ; arrêt.")
    return g


def altitude_grille():
    """Altitude moyenne par carré de 100 m (Copernicus GLO-30)."""
    cols, lignes = (X1 - X0) // PAS, (Y1 - Y0) // PAS
    t = from_origin(X0, Y1, PAS, PAS)
    alt = np.full((lignes, cols), np.nan, dtype="float32")
    for lat in range(int(BBOX[1]), int(np.ceil(BBOX[3]))):
        for lon in range(int(BBOX[0]), int(np.ceil(BBOX[2]))):
            n = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            with rasterio.open("/vsicurl/" + DEM.format(n=n)) as src:
                m = np.full((lignes, cols), np.nan, dtype="float32")
                reproject(rasterio.band(src, 1), m, dst_transform=t, dst_crs=CRS, dst_nodata=np.nan,
                          resampling=Resampling.average)
            alt = np.where(np.isnan(alt), m, alt)
    return alt


def lire_altitude(alt, x, y):
    c = np.clip(((np.asarray(x) - X0) // PAS).astype(int), 0, alt.shape[1] - 1)
    li = np.clip(((Y1 - np.asarray(y)) // PAS).astype(int), 0, alt.shape[0] - 1)
    return alt[li, c]


def morceaux_rivieres(tr):
    """Fusionne les tronçons contigus d'un même cours d'eau, puis découpe en morceaux d'au plus 2,5 km."""
    tr = tr.copy()
    tr["nom"] = tr["cpx_toponyme_de_cours_d_eau"].fillna("").astype(str)
    tr["ordre"] = pd.to_numeric(tr["numero_d_ordre"], errors="coerce").fillna(0).astype(int)
    tr["largeur"] = tr["classe_de_largeur"].map(LARGEUR).fillna(0).astype(int)
    tr["canal"] = (tr["nature"] != "Ecoulement naturel").astype(int)
    lignes = []
    for (nom, ordre, largeur, canal), g in tr.groupby(["nom", "ordre", "largeur", "canal"]):
        geoms = [force_2d(x) for x in g.geometry if x is not None and not x.is_empty]
        fusion = linemerge(geoms) if len(geoms) > 1 else geoms[0]
        parts = list(fusion.geoms) if isinstance(fusion, MultiLineString) else [fusion]
        for ligne in parts:
            n = max(1, int(np.ceil(ligne.length / LONGUEUR_MAX)))
            pas = ligne.length / n
            for k in range(n):
                seg = substring(ligne, k * pas, (k + 1) * pas)
                if isinstance(seg, LineString) and seg.length > 50:
                    lignes.append({"nom": nom, "ordre": ordre, "largeur": largeur, "canal": canal, "geometry": seg})
    return gpd.GeoDataFrame(lignes, geometry="geometry", crs=CRS)


def lieux_de_peche(emprise):
    """Lieux de pêche signalés dans OpenStreetMap (points), avec mention éventuelle d'une AAPPMA."""
    lon1, lat1, lon2, lat2 = emprise
    points = []
    for lo in np.arange(lon1, lon2, 1.0):
        for la in np.arange(lat1, lat2, 1.0):
            q = (f'[out:json][timeout:90];(nwr["leisure"="fishing"]({la},{lo},{min(la + 1, lat2)},{min(lo + 1, lon2)});'
                 f'nwr["fishing"]["fishing"!="no"]({la},{lo},{min(la + 1, lat2)},{min(lo + 1, lon2)}););out tags center;')
            try:
                elements = get(OVERPASS, methode="POST", data={"data": q}, timeout=180).json().get("elements", [])
            except Exception as e:  # OpenStreetMap n'est qu'un complément
                print(f"  OpenStreetMap indisponible pour une tuile : {e}")
                continue
            for e in elements:
                c = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
                if c.get("lat") is None:
                    continue
                texte = " ".join(str(v) for v in (e.get("tags") or {}).values())
                aappma = any(m in texte.upper() for m in ["AAPPMA", "A.A.P.P.M.A", "APPMA", "ASSOCIATION DE PÊCHE",
                                                          "ASSOCIATION DE PECHE", "SOCIÉTÉ DE PÊCHE", "AMICALE DES PÊCHEURS"])
                points.append({"aappma": int(aappma), "geometry": Point(c["lon"], c["lat"])})
    if not points:
        return gpd.GeoDataFrame({"aappma": []}, geometry=[], crs=CRS)
    return gpd.GeoDataFrame(points, geometry="geometry", crs=4326).to_crs(CRS)


def main():
    DATA.mkdir(exist_ok=True)
    dep = departements()
    zone = dep.union_all()
    emprise = tuple(dep.to_crs(4326).total_bounds.round(3))

    # Plans d'eau d'abord : ils servent à écarter les axes fictifs qui traversent les lacs et étangs
    print("Plans d'eau (BD TOPO, surfaces en eau)...")
    sh = par_tuiles("BDTOPO_V3:surface_hydrographique", emprise, "surfaces en eau",
                    champs=["cleabs", "nature", "persistance", "cpx_toponyme_de_plan_d_eau", "liens_vers_plan_d_eau"])
    sh = sh[sh["nature"].isin(NATURES_PLANS) & (sh["persistance"] != "Intermittent")].copy()
    sh["geometry"] = sh.geometry.apply(force_2d)
    # Un même plan d'eau peut être découpé en plusieurs surfaces : on les regroupe
    sh["groupe"] = sh["liens_vers_plan_d_eau"].fillna("").astype(str)
    sh.loc[sh["groupe"] == "", "groupe"] = "s_" + sh["cleabs"].astype(str)
    sh["nom"] = sh["cpx_toponyme_de_plan_d_eau"].fillna("").astype(str)
    pl = sh.dissolve(by="groupe", aggfunc={"nature": "first", "nom": "max"}).reset_index()
    pl = pl[pl.area >= SURFACE_MIN]
    pl = pl[pl.intersects(zone)]
    pl["type"] = pl["nature"].map(NATURES_PLANS)
    pl["surface_ha"] = (pl.area / 1e4).round(2)
    print(f"  {len(pl)} plans d'eau d'au moins 0,3 ha ; {pl['type'].value_counts().to_dict()}")

    print("Cours d'eau (BD TOPO)...")
    tr = par_tuiles("BDTOPO_V3:troncon_hydrographique", emprise, "tronçons",
                    champs=["cleabs", "nature", "fictif", "persistance", "numero_d_ordre", "classe_de_largeur",
                            "cpx_toponyme_de_cours_d_eau"])
    print(f"  {len(tr)} tronçons bruts")
    nomme = tr["cpx_toponyme_de_cours_d_eau"].notna()
    ordre = pd.to_numeric(tr["numero_d_ordre"], errors="coerce").fillna(0)
    garder = (tr["nature"].isin(NATURES_COURS) & (tr["persistance"] == "Permanent")
              & ((ordre >= 2) | nomme | (tr["nature"] != "Ecoulement naturel")))
    tr = tr[garder]
    tr = tr[tr.intersects(zone)]
    # Les grandes rivières et les canaux (Rhin, Ill, Moselle, Meuse, Sarre, canaux de navigation...) sont dessinés
    # comme des surfaces en eau : leur axe est un tronçon « fictif ». On le garde, sauf à l'intérieur des lacs et
    # étangs, déjà comptés comme plans d'eau.
    fictif = tr["fictif"].astype(str).str.lower().isin(["true", "1"])
    fic = tr[fictif]
    if len(fic) and len(pl):
        j = gpd.sjoin(fic[["geometry"]], pl[["geometry"]], how="inner", predicate="intersects")
        cand = fic.loc[j.index.unique()]
        surfaces = pl.geometry.union_all()
        dans = cand.geometry.intersection(surfaces).length / cand.length.clip(lower=1)
        tr = tr.drop(dans[dans > 0.5].index)
    print(f"  {len(tr)} tronçons permanents retenus, dont {int(tr['fictif'].astype(str).str.lower().isin(['true', '1']).sum())} "
          "axes de grandes rivières ou canaux")
    riv = morceaux_rivieres(tr)
    riv["type"] = np.where(riv["canal"] == 1, "canal", "riviere")
    rhin = riv["nom"].str.contains(r"^(?:le |la )?(?:Vieux |Petit )?Rhin$|Grand Canal d'Alsace", case=False, regex=True)
    riv.loc[rhin & (riv["largeur"] >= 3), "type"] = "rhin"
    print(f"  {len(riv)} morceaux de rivière ; {riv['type'].value_counts().to_dict()}")

    print("Altitude (Copernicus GLO-30)...")
    alt = altitude_grille()
    debut = riv.geometry.apply(lambda g: g.coords[0])
    fin = riv.geometry.apply(lambda g: g.coords[-1])
    z1 = lire_altitude(alt, [p[0] for p in debut], [p[1] for p in debut])
    z2 = lire_altitude(alt, [p[0] for p in fin], [p[1] for p in fin])
    riv["altitude"] = np.round((z1 + z2) / 2)
    riv["pente_pm"] = np.round(np.abs(z1 - z2) / riv.length * 1000, 2)   # pour mille
    riv["longueur_m"] = riv.length.round()
    c = pl.geometry.representative_point()
    pl["altitude"] = np.round(lire_altitude(alt, c.x, c.y))

    print("Lieux de pêche (OpenStreetMap)...")
    lp = lieux_de_peche(emprise)
    print(f"  {len(lp)} lieux signalés, dont {int(lp['aappma'].sum()) if len(lp) else 0} mentionnant une association")

    eaux = pd.concat([
        riv[["type", "nom", "ordre", "largeur", "altitude", "pente_pm", "longueur_m", "geometry"]],
        pl[["type", "nom", "surface_ha", "altitude", "geometry"]],
    ], ignore_index=True)
    eaux = gpd.GeoDataFrame(eaux, geometry="geometry", crs=CRS)
    eaux["peche"], eaux["aappma"] = 0, 0
    if len(lp):
        proche = gpd.sjoin_nearest(lp, eaux[["geometry"]], how="left", max_distance=150)
        proche = proche.dropna(subset=["index_right"])
        for idx, g in proche.groupby("index_right"):
            eaux.loc[int(idx), "peche"] = 1
            eaux.loc[int(idx), "aappma"] = int(g["aappma"].max())
    eaux["geometry"] = eaux.geometry.simplify(5)
    eaux = eaux.reset_index(drop=True)
    eaux["id"] = np.arange(len(eaux))
    for col in ["ordre", "largeur", "longueur_m", "surface_ha"]:
        eaux[col] = eaux[col].fillna(0)
    eaux.to_file(DATA / "eaux.gpkg", driver="GPKG")
    print(eaux["type"].value_counts().to_dict())
    print(f"{int(eaux.peche.sum())} eaux avec un lieu de pêche signalé, dont {int(eaux.aappma.sum())} liées à une association")


if __name__ == "__main__":
    main()
