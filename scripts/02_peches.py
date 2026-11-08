"""Étape 2 : pêches scientifiques de l'OFB (Hub'Eau « Poissons »), présence et absence des espèces par station.

Chaque pêche électrique recense toutes les espèces présentes : une station pêchée où une espèce
n'a jamais été vue est une vraie absence, ce qui permet d'apprendre où vit chaque poisson.

Résultat : data/peches.csv (une ligne par station)
"""
import pandas as pd

from commun import DATA, DEPARTEMENTS, ESPECES, get

API = "https://hubeau.eaufrance.fr/api/v1/etat_piscicole"
ANNEE_MIN = 2005
MAX_LIGNES = 20000


def requete(point, params, champs):
    """Toutes les lignes d'une requête Hub'Eau, découpée par période si elle dépasse la limite."""
    base = dict(params, fields=",".join(champs), date_operation_min=f"{ANNEE_MIN}-01-01")
    premier = get(f"{API}/{point}", dict(base, size=1)).json()
    total = premier.get("count", 0)
    if total == 0:
        return pd.DataFrame(columns=champs)
    if total <= MAX_LIGNES:
        return pd.DataFrame(get(f"{API}/{point}", dict(base, size=MAX_LIGNES), timeout=300).json()["data"])
    morceaux = []
    for annee in range(ANNEE_MIN, pd.Timestamp.today().year + 1):
        p = dict(base, date_operation_min=f"{annee}-01-01", date_operation_max=f"{annee}-12-31", size=MAX_LIGNES)
        morceaux.append(pd.DataFrame(get(f"{API}/{point}", p, timeout=300).json()["data"]))
    return pd.concat(morceaux, ignore_index=True)


def main():
    DATA.mkdir(exist_ok=True)
    champs_op = ["code_operation", "date_operation", "code_station", "libelle_entite_hydrographique",
                 "coordonnee_x_point_prelevement", "coordonnee_y_point_prelevement",
                 "coordonnee_x_station", "coordonnee_y_station", "code_epsg_projection_station"]
    ops = pd.concat([requete("operations", {"code_departement": d}, champs_op) for d in DEPARTEMENTS], ignore_index=True)
    ops["code_operation"] = pd.to_numeric(ops["code_operation"], errors="coerce").astype("Int64").astype(str)
    ops = ops.drop_duplicates("code_operation")
    ops["x"] = ops["coordonnee_x_point_prelevement"].fillna(ops["coordonnee_x_station"])
    ops["y"] = ops["coordonnee_y_point_prelevement"].fillna(ops["coordonnee_y_station"])
    ops["annee"] = pd.to_datetime(ops["date_operation"]).dt.year
    print(f"{len(ops)} pêches scientifiques depuis {ANNEE_MIN}, sur {ops.code_station.nunique()} stations")

    stations = ops.groupby("code_station").agg(
        x=("x", "median"), y=("y", "median"), riviere=("libelle_entite_hydrographique", "first"),
        n_peches=("code_operation", "nunique"), premiere=("annee", "min"), derniere=("annee", "max")).reset_index()

    for espece, infos in ESPECES.items():
        obs = []
        for code in infos["codes"]:
            for d in DEPARTEMENTS:
                o = requete("observations", {"code_departement": d, "code_alternatif_taxon": code},
                            ["code_operation", "code_station", "effectif_lot"])
                if len(o):
                    obs.append(o)
        if obs:
            o = pd.concat(obs, ignore_index=True)
            o["code_operation"] = pd.to_numeric(o["code_operation"], errors="coerce").astype("Int64").astype(str)
            o = o[pd.to_numeric(o["effectif_lot"], errors="coerce").fillna(1) > 0]
            o = o.merge(ops[["code_operation", "annee"]], on="code_operation", how="inner")
            par_station = o.groupby("code_station").agg(n=("code_operation", "nunique"), der=("annee", "max"))
            stations[f"{espece}_n"] = stations["code_station"].map(par_station["n"]).fillna(0).astype(int)
            stations[f"{espece}_derniere"] = stations["code_station"].map(par_station["der"]).fillna(0).astype(int)
        else:
            stations[f"{espece}_n"], stations[f"{espece}_derniere"] = 0, 0
        n_pres = int((stations[f"{espece}_n"] > 0).sum())
        print(f"{infos['nom']} : présent sur {n_pres} stations sur {len(stations)}")

    stations.to_csv(DATA / "peches.csv", index=False)


if __name__ == "__main__":
    main()
