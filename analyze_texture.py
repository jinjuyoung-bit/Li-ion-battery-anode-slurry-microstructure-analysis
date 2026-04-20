"""
analyze_texture.py
GLCM 기반 텍스처 특성 분석 — MATLAB analyze_texture_from_image2.m 의 Python 변환
glcm_features.py 와 함께 사용하세요.
"""

import numpy as np
import pandas as pd
from skimage import io, color
from skimage.feature import graycomatrix
from skimage.util import img_as_ubyte
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys

from glcm_features import GLCM_Features1


def analyze_texture_from_image(file_path: str = None) -> pd.DataFrame:
    """
    이미지 파일에서 GLCM을 생성하고 방향 평균 텍스처 특성을 반환합니다.

    Parameters
    ----------
    file_path : str
        분석할 이미지 파일 경로

    Returns
    -------
    pd.DataFrame
        텍스처 특성별 방향 평균값 (index=feature name, column='Mean')
    """

    if file_path is None:
        file_path = r"F:\EMCCD\250508\SG\2min\7\AVG_.tif"

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 이미지 로드 → 그레이스케일 → uint8
    image = io.imread(file_path)
    if image.ndim == 3:
        image = color.rgb2gray(image)
    if image.dtype != np.uint8:
        image = img_as_ubyte(image)

    # GLCM 생성 (MATLAB offsets: [0 1; -1 1; -1 0; -1 -1] → 4방향)
    distances = [1]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=False          # GLCM_Features1 내부에서 정규화
    )
    # shape: (256, 256, 1, 4) → (256, 256, 4) 로 squeeze
    glcm = glcm[:, :, 0, :]  # (L, L, K=4)

    # 특성 추출
    raw_stats = GLCM_Features1(glcm, makeSymmetric=1)

    # 방향 평균
    avg_values = {feat: float(np.mean(vals))
                  for feat, vals in raw_stats.items()}

    df = pd.DataFrame.from_dict(avg_values, orient="index", columns=["Mean"])
    df.index.name = "Feature"

    print("GLCM 평균 텍스처 특성 분석 결과:")
    print(df.round(6))
    return df


def compare_groups(file_paths_dict: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    그룹별(흑연 종류 등) 텍스처 특성을 비교하고 One-way ANOVA p-value를 반환합니다.

    Parameters
    ----------
    file_paths_dict : dict
        {"그룹명": [경로1, 경로2, ...], ...}
    alpha : float
        유의수준

    Returns
    -------
    pd.DataFrame
        특성별 그룹 평균 ± std 및 F통계량, p-value
    """

    group_data = {}
    for group_name, paths in file_paths_dict.items():
        records = []
        for p in paths:
            try:
                df = analyze_texture_from_image(p)
                records.append(df["Mean"].to_dict())
            except Exception as e:
                print(f"  건너뜀: {p} ({e})")
        if records:
            group_data[group_name] = pd.DataFrame(records)

    if not group_data:
        raise ValueError("분석 가능한 이미지가 없습니다.")

    features = list(next(iter(group_data.values())).columns)
    rows = []

    for feat in features:
        row = {"Feature": feat}
        arrays = []
        for gname, gdf in group_data.items():
            arr = gdf[feat].values
            arrays.append(arr)
            row[f"{gname}_mean"] = round(arr.mean(), 6)
            row[f"{gname}_std"]  = round(arr.std(),  6)

        if len(arrays) >= 2 and all(len(a) >= 2 for a in arrays):
            f_stat, p_val = stats.f_oneway(*arrays)
            row["F_statistic"] = round(f_stat, 4)
            row["p_value"]     = round(p_val,  6)
            row["significant"] = "Yes" if p_val < alpha else "No"
        else:
            row["F_statistic"] = None
            row["p_value"]     = None
            row["significant"] = "N/A"

        rows.append(row)

    result_df = pd.DataFrame(rows).set_index("Feature")
    print("\n그룹 비교 결과 (One-way ANOVA):")
    print(result_df)
    return result_df


def plot_texture_comparison(file_paths_dict: dict, save_path: str = None):
    """
    그룹별 텍스처 특성 boxplot 시각화
    """

    group_data = {}
    for gname, paths in file_paths_dict.items():
        records = []
        for p in paths:
            try:
                df = analyze_texture_from_image(p)
                records.append(df["Mean"].to_dict())
            except Exception:
                pass
        if records:
            group_data[gname] = pd.DataFrame(records)

    features = list(next(iter(group_data.values())).columns)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data   = [gdf[feat].values for gdf in group_data.values()]
        labels = list(group_data.keys())
        ax.boxplot(data, labels=labels)
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("Mean")
        ax.tick_params(axis="x", rotation=15)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("GLCM Texture Feature Comparison by Graphite Type",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장 완료: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_texture_from_image(sys.argv[1])
    else:
        print("사용법: python analyze_texture.py <이미지_경로>")
        print("예시 (그룹 비교):")
        print("""
from analyze_texture import compare_groups, plot_texture_comparison

groups = {
    "천연흑연": ["./data/NG_01.tif", "./data/NG_02.tif"],
    "인조흑연": ["./data/AG_01.tif", "./data/AG_02.tif"],
    "볼밀흑연": ["./data/BM_01.tif", "./data/BM_02.tif"],
}
compare_groups(groups)
plot_texture_comparison(groups, save_path="./results/comparison.png")
        """)
