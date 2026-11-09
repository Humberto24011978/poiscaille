"""Étape 2 bis : accès aux eaux (domaine public, lots d'AAPPMA, réciprocités, réserves, privé).

Sources :
- tronçons navigables de la BD TOPO (IGN) : Rhin, canaux et rivières canalisées, qui relèvent du domaine public fluvial ;
- carte des parcours publiée par la fédération de pêche du Haut-Rhin (Google My Maps) : AAPPMA, catégorie,
  réciprocité, domaine public ou fédéral, réserves, parcours spécifiques, étangs hors URNE ;
- lieux de pêche d'OpenStreetMap : nom de l'association, étang privé ou payant ;
- limites départementales (ADMIN EXPRESS), pour le lien vers la carte Géopêche de chaque fédération.

Statut de chaque eau : public, aappma, prive, inconnu.

Résultat : data/acces.csv
"""
import re
import unicodedata
import xml.etree.ElementTree as ET
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from commun import BBOX, CRS, DATA, DEPARTEMENTS, get

WFS = "https://data.geopf.fr/wfs/ows"
OVERPASS = "https://overpass-api.de/api/interpreter"
KML_68 = "https://www.google.com/maps/d/kml"
CARTE_68 = "1XIAIW73_lXAKp7NfJY9gEgytV84scjQB"
NS = {"k": "http://www.opengis.net/kml/2.2"}
TAMPON_LIGNE = 40        # m : distance maximale entre un parcours de la fédération et notre morceau de rivière
PART_MIN = 0.4           # part minimale du morceau de rivière couverte par le parcours
DIST_POINT = 300         # m : rattachement d'un point (étang, parcours spécifique) à l'eau la plus proche
RIVIERES = ["riviere", "canal", "rhin"]
MOTS_ASSOC = ["AAPPMA", "A.A.P.P.M.A", "APPMA", "ASSOCIATION DE PECHE", "SOCIETE DE PECHE", "AMICALE DES PECHEURS",
              "PECHE CLUB", "CLUB DE PECHE", "UNION DES PECHEURS", "LES PECHEURS"]


def simple(t):
    return unicodedata.normalize("NFD", str(t or "")).encode("ascii", "ignore").decode().upper()


# ---------- Domaine public : tronçons navigables ----------

def navigables():
    morceaux, debut = [], 0
    filtre = f"navigabilite=true AND BBOX(geometrie,{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]},'EPSG:4326')"
    while True:
        r = get(WFS, {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                      "typeNames": "BDTOPO_V3:troncon_hydrographique", "outputFormat": "application/json",
                      "srsName": CRS, "count": 5000, "startIndex": debut, "sortBy": "cleabs",
                      "propertyName": "cleabs,navigabilite,geometrie", "cql_filter": filtre}, timeout=300)
        g = gpd.read_file(BytesIO(r.content))
        if g.empty:
            break
        morceaux.append(g)
        if len(g) < 5000:
            break
        debut += 5000
    if not morceaux:
        return gpd.GeoDataFrame(geometry=[], crs=CRS)
    return gpd.GeoDataFrame(pd.concat(morceaux, ignore_index=True), geometry="geometry", crs=CRS)


# ---------- Carte des parcours du Haut-Rhin ----------

def lire_kml(contenu):
    """Lit la carte de la fédération : un objet par parcours, avec son dossier et ses champs."""
    racine = ET.fromstring(contenu)
    objets = []

    def coords(el):
        texte = el.findtext("k:coordinates", default="", namespaces=NS).split()
        return [tuple(float(v) for v in c.split(",")[:2]) for c in texte if c.count(",") >= 1]

    for dossier in racine.iter(f"{{{NS['k']}}}Folder"):
        nom_dossier = dossier.findtext("k:name", default="", namespaces=NS).strip()
        for pm in dossier.findall("k:Placemark", NS):
            champs = {d.get("name"): (d.findtext("k:value", default="", namespaces=NS) or "").strip()
                      for d in pm.iter(f"{{{NS['k']}}}Data")}
            geoms = []
            for ls in pm.iter(f"{{{NS['k']}}}LineString"):
                c = coords(ls)
                if len(c) >= 2:
                    geoms.append(LineString(c))
            for pg in pm.iter(f"{{{NS['k']}}}Polygon"):
                ext = pg.find("k:outerBoundaryIs/k:LinearRing", NS)
                if ext is not None:
                    c = coords(ext)
                    if len(c) >= 4:
                        geoms.append(Polygon(c))
            for pt in pm.iter(f"{{{NS['k']}}}Point"):
                c = coords(pt)
                if c:
                    geoms.append(Point(c[0]))
            for g in geoms:
                objets.append({"dossier": nom_dossier, "nom": pm.findtext("k:name", default="", namespaces=NS).strip(),
                               "aappma": champs.get("AAPPMA", ""), "cat_recip": champs.get("Catégorie et réciprocité", ""),
                               "parcours": champs.get("Parcours", ""), "reglement": champs.get("Lien vers le règlement", ""),
                               "poissons": champs.get("Population piscicole", ""), "geometry": g})
    if not objets:
        return gpd.GeoDataFrame(geometry=[], crs=CRS)
    return gpd.GeoDataFrame(objets, geometry="geometry", crs=4326).to_crs(CRS)


def classer(row):
    """Catégorie (1/2), statut (public/aappma), réciprocité, réserve, à partir du dossier et des champs."""
    texte = simple(row["dossier"] + " " + row["cat_recip"])
    cat = 1 if re.search(r"\b1(ERE|RE|E)?\b|PREMIERE", texte) else 2 if re.search(r"\b2(EME|E)?\b|DEUXIEME", texte) else 0
    reserve = "RESERVE" in texte
    specifique = "SPECIFIQUE" in texte
    depositaire = "DEPOSITAIRE" in texte or "POINT DE VENTE" in texte
    hors_urne = "HORS URNE" in texte
    if "NON RECIPROC" in texte or "NON-RECIPROC" in texte:
        recip = "non réciprocitaire"
    elif hors_urne:
        recip = "hors URNE (carte de l'AAPPMA nécessaire)"
    elif "RECIPROC" in texte:
        recip = "réciprocitaire (URNE)"
    else:
        recip = ""
    public = ("DOMAINE PUBLIC" in texte or "FEDERAL" in texte) and not row["aappma"]
    statut = "public" if public else "aappma" if (row["aappma"] or "RECIPROC" in texte or hors_urne) else ""
    return pd.Series({"cat": cat, "statut": statut, "recip": recip, "reserve": reserve,
                      "specifique": specifique, "depositaire": depositaire})


def rattacher_federation(eaux, fed):
    """Associe chaque objet de la carte fédérale aux eaux correspondantes."""
    res = {}
    if fed.empty:
        return res
    fed = pd.concat([fed, fed.apply(classer, axis=1)], axis=1)
    fed = fed[~fed["depositaire"]]
    riv = eaux[eaux["type"].isin(RIVIERES)]
    pl = eaux[~eaux["type"].isin(RIVIERES)]

    def ajouter(idx, row, force=False, part=1.0):
        info = res.setdefault(int(idx), {})
        if row["reserve"]:
            # 1 : réserve sur (presque) tout le morceau ; 2 : réserve sur une partie seulement
            niveau = 1 if part >= 0.6 else 2
            info["reserve"] = min(info.get("reserve", 2), niveau)
            return
        if row["specifique"]:
            info["specifique"] = row["parcours"] or row["nom"]
            if row["reglement"]:
                info.setdefault("reglement", row["reglement"])
            return
        if force or not info.get("statut"):
            for k in ["statut", "recip"]:
                if row[k]:
                    info[k] = row[k]
            if row["cat"]:
                info["cat"] = int(row["cat"])
            if row["aappma"]:
                info["aappma"] = row["aappma"]
            if row["reglement"]:
                info["reglement"] = row["reglement"]
            info["source"] = "fédération du Haut-Rhin"

    lignes = fed[fed.geom_type == "LineString"]
    if len(lignes) and len(riv):
        tampons = lignes.copy()
        tampons["geometry"] = lignes.buffer(TAMPON_LIGNE)
        j = gpd.sjoin(riv[["geometry"]], tampons, how="inner", predicate="intersects")
        for idx_riv, g in j.groupby(level=0):
            seg = riv.geometry.loc[idx_riv]
            for _, row in g.iterrows():
                part = seg.intersection(tampons.geometry.loc[row["index_right"]]).length / max(seg.length, 1)
                if (row["reserve"] and part >= 0.02) or part >= PART_MIN:
                    ajouter(idx_riv, row, part=part)
    polys = fed[fed.geom_type == "Polygon"]
    if len(polys):
        j = gpd.sjoin(eaux[["geometry"]], polys, how="inner", predicate="intersects")
        for idx, g in j.groupby(level=0):
            geom = eaux.geometry.loc[idx]
            for _, row in g.iterrows():
                inter = geom.intersection(polys.geometry.loc[row["index_right"]])
                part = (inter.area / max(geom.area, 1)) if geom.area > 0 else (inter.length / max(geom.length, 1))
                if part >= 0.3:
                    ajouter(idx, row, force=True)
    points = fed[fed.geom_type == "Point"]
    if len(points):
        cibles = pd.concat([pl[["geometry"]], riv[["geometry"]]])
        j = gpd.sjoin_nearest(points, cibles, how="inner", max_distance=DIST_POINT, distance_col="dist")
        j = j.sort_values("dist").drop_duplicates(subset=["geometry"])
        for _, row in j.iterrows():
            ajouter(row["index_right"], row, force=not row["specifique"])
    return res


# ---------- OpenStreetMap : associations, étangs privés ou payants ----------

def osm():
    lon1, lat1, lon2, lat2 = BBOX
    points = []
    for lo in np.arange(lon1, lon2, 1.0):
        for la in np.arange(lat1, lat2, 1.0):
            b = f"{la},{lo},{min(la + 1, lat2)},{min(lo + 1, lon2)}"
            q = (f'[out:json][timeout:90];(nwr["leisure"="fishing"]({b});nwr["fishing"]["fishing"!="no"]({b}););'
                 'out tags center;')
            try:
                elements = get(OVERPASS, methode="POST", data={"data": q}, timeout=180).json().get("elements", [])
            except Exception as e:
                print(f"  OpenStreetMap indisponible pour une tuile : {e}")
                continue
            for e in elements:
                c = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
                if c.get("lat") is None:
                    continue
                t = e.get("tags") or {}
                assoc = ""
                for cle in ["operator", "name", "owner", "description"]:
                    if any(m in simple(t.get(cle)) for m in MOTS_ASSOC):
                        assoc = t[cle]
                        break
                acces = (t.get("access") or "").lower()
                prive = acces in ("private", "no") or (t.get("fee") == "yes" and not assoc)
                points.append({"assoc": assoc, "prive": int(prive), "payant": int(t.get("fee") == "yes"),
                               "geometry": Point(c["lon"], c["lat"])})
    if not points:
        return gpd.GeoDataFrame({"assoc": [], "prive": [], "payant": []}, geometry=[], crs=CRS)
    return gpd.GeoDataFrame(points, geometry="geometry", crs=4326).to_crs(CRS)


def departements():
    r = get(WFS, {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                  "typeNames": "ADMINEXPRESS-COG-CARTO.LATEST:departement", "outputFormat": "application/json",
                  "srsName": CRS, "count": 100,
                  "bbox": f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]},EPSG:4326"})
    g = gpd.read_file(BytesIO(r.content))
    return g[g["code_insee"].astype(str).isin(DEPARTEMENTS)][["code_insee", "geometry"]]


def main():
    eaux = gpd.read_file(DATA / "eaux.gpkg")
    out = pd.DataFrame({"id": eaux["id"], "dep": "", "statut": "inconnu", "cat": 0, "aappma": "", "recip": "",
                        "reserve": 0, "specifique": "", "reglement": "", "source": ""}).set_index(eaux.index)

    print("Départements...")
    dep = departements()
    c = gpd.GeoDataFrame(geometry=eaux.geometry.representative_point(), crs=CRS)
    j = gpd.sjoin_nearest(c, dep, how="left", max_distance=5000)
    j = j[~j.index.duplicated()]
    out["dep"] = j["code_insee"].reindex(eaux.index).fillna("").astype(str)

    print("Domaine public (tronçons navigables de la BD TOPO)...")
    nav = navigables()
    print(f"  {len(nav)} tronçons navigables")
    riv = eaux[eaux["type"].isin(RIVIERES)]
    if len(nav):
        tampon = nav.buffer(25).union_all()
        part = riv.geometry.intersection(tampon).length / riv.length.clip(lower=1)
        out.loc[part[part >= 0.5].index, ["statut", "source"]] = ["public", "domaine public fluvial (IGN)"]
    out.loc[eaux["type"].isin(["rhin"]), ["statut", "source"]] = ["public", "domaine public fluvial (IGN)"]

    print("Carte des parcours de la fédération du Haut-Rhin...")
    try:
        kml = get(KML_68, {"mid": CARTE_68, "forcekml": 1}, timeout=120).content
        fed = lire_kml(kml)
        print(f"  {len(fed)} objets lus")
        res = rattacher_federation(eaux, fed)
    except Exception as e:  # la carte de la fédération n'est qu'un complément
        print(f"  carte indisponible : {e}")
        res = {}
    for idx, info in res.items():
        for k, v in info.items():
            out.loc[idx, k] = v
    n68 = sum(1 for v in res.values() if v.get("statut"))
    print(f"  {n68} eaux rattachées à un parcours, {sum(1 for v in res.values() if v.get('reserve'))} en réserve, "
          f"{sum(1 for v in res.values() if v.get('specifique'))} parcours spécifiques")

    print("Lieux de pêche (OpenStreetMap)...")
    lp = osm()
    print(f"  {len(lp)} lieux, {int((lp['assoc'] != '').sum()) if len(lp) else 0} avec une association, "
          f"{int(lp['prive'].sum()) if len(lp) else 0} privés ou payants")
    if len(lp):
        j = gpd.sjoin_nearest(lp, eaux[["geometry"]], how="inner", max_distance=150, distance_col="dist")
        for _, row in j.sort_values("dist").iterrows():
            idx = row["index_right"]
            if out.loc[idx, "source"] == "fédération du Haut-Rhin":
                continue
            if row["assoc"] and out.loc[idx, "statut"] in ("inconnu", "prive"):
                out.loc[idx, ["statut", "aappma", "source"]] = ["aappma", row["assoc"], "OpenStreetMap"]
            elif row["prive"] and out.loc[idx, "statut"] == "inconnu":
                out.loc[idx, ["statut", "source"]] = ["prive", "OpenStreetMap (privé ou payant)"]

    out.to_csv(DATA / "acces.csv", index=False)
    print("Statuts :", out["statut"].value_counts().to_dict())


if __name__ == "__main__":
    main()
