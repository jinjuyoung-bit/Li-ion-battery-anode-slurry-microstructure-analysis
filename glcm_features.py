"""
glcm_features.py
MATLAB GLCM_Features1.m 의 1:1 Python 변환
"""

import numpy as np


def GLCM_Features1(glcmin: np.ndarray, makeSymmetric: int = 0) -> dict:
    """
    GLCM에서 11가지 텍스처 특성을 추출합니다.
    MATLAB GLCM_Features1.m 과 동일한 계산 로직.

    Parameters
    ----------
    glcmin : np.ndarray
        shape (L, L) 또는 (L, L, K) — 카운트값 또는 확률 모두 가능
    makeSymmetric : int
        1이면 P + P.T 로 대칭화, 0이면 그대로

    Returns
    -------
    dict
        키: 'autoc','contr','corrm','corrp','energ','entro',
             'homop','maxpr','sosvh','senth','denth'
        값: shape (K,) ndarray
    """

    # 2D → 3D 통일
    if glcmin.ndim == 2:
        glcmin = glcmin[:, :, np.newaxis]

    L, _, K = glcmin.shape

    out_fields = ['autoc', 'contr', 'corrm', 'corrp', 'energ',
                  'entro', 'homop', 'maxpr', 'sosvh', 'senth', 'denth']
    out = {f: np.full(K, np.nan) for f in out_fields}

    # 인덱스 격자 (1-based, MATLAB 동일)
    j_idx, i_idx = np.meshgrid(np.arange(1, L+1), np.arange(1, L+1))
    # i_idx[r,c] = r+1, j_idx[r,c] = c+1

    for k in range(K):
        P = glcmin[:, :, k].astype(float)

        if makeSymmetric:
            P = P + P.T

        s = P.sum()
        if s == 0:
            P = np.ones((L, L)) / (L * L)
        else:
            P = P / s

        # 한계분포
        px = P.sum(axis=1)          # (L,) 행 합
        py = P.sum(axis=0)          # (L,) 열 합

        ii = np.arange(1, L+1, dtype=float)

        # 평균 / 표준편차
        ux = np.sum(ii * px)
        uy = np.sum(ii * py)
        sx = np.sqrt(np.sum(((ii - ux)**2) * px) + np.finfo(float).eps)
        sy = np.sqrt(np.sum(((ii - uy)**2) * py) + np.finfo(float).eps)

        # --- 11가지 특성 ---
        out['autoc'][k] = np.sum(i_idx * j_idx * P)

        out['contr'][k] = np.sum((i_idx - j_idx)**2 * P)

        out['corrm'][k] = (np.sum((i_idx - ux) * (j_idx - uy) * P)
                           / (sx * sy))

        out['corrp'][k] = ((np.sum(i_idx * j_idx * P) - ux * uy)
                           / (sx * sy))

        out['energ'][k] = np.sum(P**2)

        out['entro'][k] = -np.sum(P * np.log(P + np.finfo(float).eps))

        out['homop'][k] = np.sum(P / (1 + np.abs(i_idx - j_idx)))

        out['maxpr'][k] = P.max()

        out['sosvh'][k] = np.sum((i_idx - ux)**2 * P)

        # p_{x+y}: 대각선 합 (길이 2L-1)
        p_xpy = np.zeros(2 * L - 1)
        for d in range(-(L-1), L):
            p_xpy[d + (L-1)] = np.trace(P, offset=d)
        out['senth'][k] = -np.sum(p_xpy * np.log(p_xpy + np.finfo(float).eps))

        # p_{|x-y|}: 차이 히스토그램 (길이 L)
        p_xmy = np.zeros(L)
        for d in range(L):
            p_xmy[d] = P[np.abs(i_idx - j_idx) == d].sum()
        out['denth'][k] = -np.sum(p_xmy * np.log(p_xmy + np.finfo(float).eps))

    return out
