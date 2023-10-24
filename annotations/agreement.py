#!/usr/bin/env python3

from collections import Counter
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd


def distNumeric(l1: float, l2: float) -> float:
    return float(np.abs(l1 - l2))


def computePairwiseAgreement(
    df: pd.DataFrame,
    valCol: str,
    groupCol: str = "HITId",
    minN: int = 2,
    distF: Callable[[float, float], float] = distNumeric,
) -> tuple[float, int, pd.Series]:  # type: ignore[type-arg]
    """Computes pairwise agreement.
    valCol: the column with the answers (e.g., Lickert scale values)
    groupCol: the column identifying the rated item (e.g., HITId, post Id, etc)
    """
    g = df.groupby(groupCol)[valCol]
    ppas = {}
    n = 0
    for s, votes in g:
        if len(votes) >= minN:
            pa = np.mean([1 - distF(*v) for v in combinations(votes, r=2)])
            ppas[s] = pa
            n += 1
            if pd.isnull(pa):  # type: ignore
                print("Pairwise agreement is null for group: ")
                print(g)
                # embed()
        # else: print(len(votes))
    if len(ppas) == 0:
        return np.nan, n, pd.Series(ppas)
    else:
        ppa = float(np.mean(list(ppas.values())))
        if pd.isnull(ppa):
            print(f"Pairwise agreement probs for column {valCol}")
            # embed()

    return ppa, n, pd.Series(ppas)


def computeRandomAgreement(
    df: pd.DataFrame,
    valCol: str,
    distF: Callable[[float, float], float] = distNumeric,
) -> float:
    distrib = Counter(df[valCol])
    agree = 0.0
    tot = 0.0
    i = 0
    for p1 in distrib:
        for p2 in distrib:
            a1 = p1
            a2 = p2
            num, denom = 1 - distF(a1, a2), 1
            if p1 == p2:
                agree += distrib[p1] * (distrib[p2] - 1) * num / denom
                tot += distrib[p1] * (distrib[p2] - 1)
            else:
                agree += distrib[p1] * (distrib[p2]) * num / denom
                tot += distrib[p1] * distrib[p2]
            i += 1
    return agree / tot


def create_fleiss_table(
    df: pd.DataFrame, col: str, groupCol: str
) -> pd.DataFrame:
    # Group the data by the group column and count ratings per category
    fleiss_df = df.groupby([groupCol, col])[col].count().unstack(fill_value=0)
    # Convert to a numpy array and add a row representing total ratings per group
    fleiss_table = fleiss_df.to_numpy()
    return pd.DataFrame(fleiss_table, columns=range(fleiss_table.shape[1]))


def fleiss_kappa(
    df: pd.DataFrame, n_rater: int, method: str = "fleiss"
) -> float:
    df = df.copy()
    n_categories = df.shape[1]

    table = df.to_numpy()
    # Calculate observed agreement
    sum_rater = table.sum(axis=1)
    # filter out rows with not enough ratings
    table = table[sum_rater >= n_rater]
    n_sub = table.shape[0]
    p_mean = ((table**2).sum(axis=1) - n_rater).sum() / (
        n_rater * (n_rater - 1) * n_sub
    )
    if method == "fleiss":
        p_mean_exp = ((table.sum(axis=0) ** 2).sum()) / (
            n_sub**2 * n_rater**2
        )
    elif method.startswith("rand") or method.startswith("unif"):
        p_mean_exp = 1 / n_categories
    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return float(kappa)


def computeFleissKappa(
    df: pd.DataFrame,
    col: str,
    groupCol: str,
    n_rater: int,
    method: str = "randolf",
) -> float:
    df = df.copy()
    df = df[[groupCol, col]]
    # Calculate the sum of squared ratings per category
    fleiss_table = create_fleiss_table(df, col, groupCol)

    # Calculate Fleiss' Kappa using the modified function
    score = fleiss_kappa(fleiss_table, n_rater, method=method)
    return score


def computeAlpha(
    df: pd.DataFrame,
    valCol: str,
    groupCol: str = "HITId",
    minN: int = 2,
    distF: Callable[[float, float], float] = distNumeric,
) -> dict[str, float]:
    """Computes Krippendorf's Alpha"""
    d = df[~df[valCol].isnull()]
    ppa, n, groups = computePairwiseAgreement(
        d, valCol, groupCol=groupCol, minN=minN, distF=distF
    )

    d2 = d[d[groupCol].isin(groups.index)]

    # Only computing random agreement on HITs that
    # we computed pairwise agreement for.
    if len(groups):
        rnd = computeRandomAgreement(d2, valCol, distF=distF)

        # Skew: computes how skewed the answers are; Krippendorf's Alpha
        # behaves terribly under skewed distributions.
        if d2[valCol].dtype == float or d2[valCol].dtype == int:
            skew = d2[valCol].mean()
        else:
            if isinstance(d2[valCol].iloc[0], list) or isinstance(
                d2[valCol].iloc[0], set
            ):
                skew = 0
            else:
                skew = d2[valCol].describe()["freq"] / len(d2)
    else:
        rnd = np.nan
        skew = 0
    if rnd == 1:
        alpha = np.nan
    else:
        alpha = 1 - ((1 - ppa) / (1 - rnd))
    return dict(alpha=alpha, ppa=ppa, rnd_ppa=rnd, skew=skew, n=n)


def inter_annotator_agreement(
    annotation_df: pd.DataFrame, value_column: str
) -> tuple[dict[str, float]]:
    annotation_df = annotation_df.copy()
    annotation_df[value_column] = (
        annotation_df[value_column] / 5
    )  # normalize the scale
    scores = computeAlpha(
        annotation_df, value_column, groupCol="HITId"
    )
    return scores


if __name__ == "__main__":
    annotation_df = pd.read_csv('./annotations/results_tier_3_n5_int.csv')
    print(inter_annotator_agreement(annotation_df, "Answer.privacyQuestion"))