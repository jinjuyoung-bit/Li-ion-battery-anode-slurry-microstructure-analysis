# Li-ion Battery Anode Slurry Microstructure Analysis

## 소개
아주대학교 석사 논문 "미세유체 채널 내 음극 슬러리의 미세구조 시각화 및 정량화"에서
사용한 GLCM 텍스처 분석 코드를 Python으로 구현했습니다.

2025년 한국유변학회 최우수 논문 구두 발표상 수상 연구입니다.

## 연구 배경
리튬이온배터리 음극 슬러리(흑연 + CMC 바인더 + 카본블랙)의
미세구조를 미세유체 채널에서 직접 촬영 후 GLCM으로 정량화합니다.
흑연 종류(천연/인조/볼밀)에 따른 슬러리 분산 균일성 차이를
이미지 텍스처 특성값으로 수치화하고 유변물성 데이터와 연결합니다.

## 파일 구조
- `glcm_features.py` : 11가지 GLCM 텍스처 특성 추출 함수
- `analyze_texture.py` : 이미지 로드 → GLCM 생성 → 그룹 비교(ANOVA) → 시각화

## 설치
pip install scikit-image scipy pandas matplotlib numpy

## 사용법
python analyze_texture.py ./data/sample.tif

## 기술 스택
- Python, MATLAB
- scikit-image, scipy, numpy, pandas, matplotlib
- GLCM (Gray Level Co-occurrence Matrix)
- One-way ANOVA (p-value 통계 검정)
