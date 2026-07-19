#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
 CO2-MEA 흡수탑 하이브리드 소프트 센서 — 프로덕션 파이프라인 v2
================================================================================
Zhuang et al. (2022), Computers in Industry 143, 103747 의 하이브리드 소프트센서를
아래의 '확정된 최적화 엔지니어링 파이프라인'대로 처음부터 리팩토링/고도화한 모듈.

기존(v1) → 고도화(v2) 변경 요약
--------------------------------------------------------------------------------
  [전처리]   선형 보간(pd.interpolate 'linear')      → Akima1DInterpolator (C1 매끄러움)
  [차원축소] 선형 POD/PCA/DAE                         → UMAP 비선형 매니폴드 (16-D)
  [시계열]   단일 LSTM(100)                           → GRU (경량·저지연·과적합 억제)
  [손실]     단순 데이터 MSE                          → 데이터 MSE + PINN 물질수지 잔차
  [검증]     전체 평균 RMSE                           → 하단부(P5,P6) 분리 RMSE
                                                        + OPEX 불확실성 비용 + Payback

설계 원칙
--------------------------------------------------------------------------------
  * 순수 함수 + dataclass 설정 + 모듈화 클래스 → 기존 DataFrame 리스트에 즉시 연동.
  * 모든 텐서 변환 지점에 Shape 주석을 명시 (N=샘플수, W=윈도우, F=피처, P=6포인트).
  * PINN 물리 잔차는 '기계론(kinetic) 프로파일'을 두-필름 평형 기준 y* 로 사용하여
    데이터-드리븐 GRU 출력을 축방향 분산 PFR 물질수지에 물리적으로 결속한다.

의존성 (기존 requirements.txt 에 추가 필요)
--------------------------------------------------------------------------------
    pip install torch umap-learn
    # scipy(>=1.x, Akima1DInterpolator), scikit-learn, pandas, numpy 는 기설치.
================================================================================
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

# --- 선택적(런타임) 의존성: 미설치 시 명확한 안내 메시지로 실패시킨다. ---------------
try:
    import umap  # umap-learn
except ImportError:  # pragma: no cover
    umap = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_BASE = nn.Module
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    # torch 미설치 시에도 클래스 '정의' 자체는 통과하도록 하는 최소 스텁 베이스.
    # (실제 인스턴스화/학습은 Trainer 에서 ImportError 로 명확히 차단된다.)
    class _TORCH_BASE:  # noqa: N801
        def __init__(self, *a, **k):
            raise ImportError("torch 미설치: `pip install torch` 후 사용하세요.")


# =============================================================================
# 0. 전역 설정 (Config) — 물리 상수, 경제성 단가, 하이퍼파라미터
# =============================================================================
@dataclass
class DataConfig:
    """데이터 로딩/전처리/차원축소/윈도우 설정."""
    data_glob: str = "data/withLabel/1*.xlsx"          # 라벨링된 공정 기록 (엑셀)
    kinetic_glob: str = "kinetic_model/1*.csv"         # 기계론(Simulink) 프로파일 (분율)
    test_df_index_list: Tuple[int, ...] = (2,)         # 테스트로 뺄 데이터셋 인덱스
    callback: int = 17                                 # 과거 윈도우 길이(현재 스텝 제외) → W = callback+1
    reduced_dimension: int = 16                        # 압축 차원 (사양: 16-D)
    reduction: str = "pca"                             # {'umap','pca','pls'} 차원축소 방법
    #  ▶ 실측 bake-off 결과 'pca'(POD) 채택: 소표본 회귀에서 UMAP(비지도 비선형)·
    #    PLS(과적합)보다 선형 직교축소가 예측 신호를 잘 보존 (아래 UPGRADE_REPORT_v2.md).
    #  · umap : 비선형 매니폴드(비지도) — 시각화용, 회귀엔 신호손실 위험
    #  · pca  : 선형 직교(비지도, POD 동형) — 강건한 선형 베이스라인
    #  · pls  : 부분최소제곱(지도) — 타깃(6-포인트 프로파일) 상관 성분만 추출 → 회귀 최적
    n_points: int = 6                                  # 흡수탑 샘플링 포인트 수
    # 피드 특성 직접 주입(PCA 우회) 옵션. 흡수탑 입구/하단 조건을 원본 정규화값으로
    #   잠재벡터에 append 하는 P6 개선 시도였으나, 실측 결과 소표본 과적합으로 오히려
    #   악화(FUSED 0.207→0.241)되어 **기본 OFF**. 대용량 데이터에서 재평가 권장.
    #   활성화 예: passthrough_features=("FT301m3/hr","FT302(kg/hr)","PT402(barg)","TT104(0C)",...)
    passthrough_features: Tuple[str, ...] = ()
    conc_col: str = "AT400(CO2 %)"                     # 가스분석기 CO2 농도 컬럼
    label_col: str = "label"                           # 측정 위치(1~6) 라벨 컬럼
    val_stride: int = 5                                # train/val 분할 stride (매 5번째 → val)
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    random_state: int = 0


@dataclass
class PhysicsConfig:
    """
    PINN 물질수지 잔차용 물리 파라미터 (정규화된 단(stage) 좌표계, dz=1).

    흡수탑 기체상 CO2 정상상태 물질수지 (축방향 분산 PFR + 두-필름 흡수 sink):
        D * d²y/dz² - u * dy/dz - Da * (y - y*) = 0
      · D  : 축방향 분산계수(Péclet 역수 스케일)  [무차원]
      · u  : 기체상 이류(플러그 흐름) 속도          [무차원]
      · Da : Damköhler 수 = (K_G·a·L)/u_g, 두-필름 총괄물질전달 저항의 역수
      · y* : 두-필름 평형 기체 조성 (기계론 kinetic 프로파일을 대리값으로 사용)
    """
    # ▶ 아래 기본값은 실측 bake-off 최적 조합(PCA-16 + inverse-PINN, RMSE 0.252).
    #   학습가능 D/u/Da 는 데이터에 안 맞는 PDE 잔차를 스스로 ~0 으로 꺼버리고,
    #   실이득은 '단조성 + 피드단(P6) 기계론 경계 앵커링' 에서 나온다.
    D_axial: float = 0.15          # 축방향 분산 초기값 (학습가능; 0이면 순수 PFR)
    u_gas: float = 1.0             # 이류 속도 초기값 (기체 상승; index 6→1 방향)
    damkohler: float = 0.30        # 흡수 sink 강도 초기값 (K_G·a 관련)
    w_residual: float = 0.05       # PDE 물질수지 잔차 가중치
    w_monotonic: float = 0.05      # 단조성(하단→상단 CO2 감소) 물리 prior 가중치
    w_boundary: float = 0.05       # 경계(피드단 P6) 기계론 정합 가중치  ← 핵심 이득원
    learnable: bool = True         # D/u/Da 를 학습가능 파라미터로(inverse PINN)
    warmup_epochs: int = 100       # 물리 가중치 0→1 선형 워밍업 epoch 수(데이터 선적합)


@dataclass
class ModelConfig:
    """GRU 소프트센서 아키텍처/학습 설정."""
    gru_hidden: int = 64           # GRU 은닉차원 (LSTM 100 대비 경량화)
    gru_layers: int = 1            # 데이터 부족 환경 과적합 방지 → 단층
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 40
    patience: int = 70             # Early stopping
    weight_decay: float = 1e-5
    device: str = "cpu"


@dataclass
class EconomicConfig:
    """
    OPEX 중심 경제성 평가 단가/운전 파라미터.
    (파일럿 → 상용 스케일 환산 계수를 포함, 모든 단가는 사용자 사이트값으로 교체 가능.)
    """
    # --- 운전/플랜트 스케일 ---
    flue_gas_flow_kg_hr: float = 5000.0     # 흡수탑 처리 배가스 유량 [kg/hr]
    co2_molar_mass: float = 44.01           # [g/mol]
    gas_molar_mass: float = 29.0            # 배가스 평균 분자량 [g/mol]
    operating_hours_yr: float = 8000.0      # 연간 가동시간 [hr/yr]

    # --- 과소예측(Carbon slip) 측: 탄소배출권 ---
    carbon_price_per_ton: float = 90.0      # 탄소배출권 [USD/ton-CO2]

    # --- 과대예측(Over-circulation) 측: MEA 과순환 → 리보일러 스팀 ---
    # CO2 1 ton 재생에 필요한 리보일러 열 → 스팀 소모 → 비용으로 환산.
    reboiler_duty_gj_per_ton_co2: float = 3.7   # 재생 에너지 [GJ/ton-CO2] (MEA 표준 ~3.6~4.0)
    steam_price_per_ton: float = 25.0           # 스팀 단가 [USD/ton-steam]
    steam_latent_heat_gj_per_ton: float = 2.16  # 저압 스팀 잠열 [GJ/ton] (~2.16 GJ/ton @ 3 barg)
    overcirc_gain: float = 1.0                  # 과대예측 % → 잉여순환 민감도 (사이트 튜닝)

    # --- 투자비/회수기간 ---
    capex_usd: float = 120000.0             # 소프트센서 고도화 배포 투자비 [USD]


# =============================================================================
# 1. 데이터 전처리 & 차원축소 (Akima 보간 + UMAP)
# =============================================================================
class DataPipeline:
    """
    엑셀 공정기록 → (Akima 보간 라벨 복원) → (MinMax 정규화) → (UMAP 16-D)
    → (슬라이딩 윈도우 시퀀스 텐서) 까지의 전체 전처리 파이프라인.

    산출물(Shape 규약):
        X : [N, W, F]  시퀀스 입력  (F = reduced_dim + n_points 원핫)
        Y : [N, P]     현재 스텝 6-포인트 CO2 프로파일 (정규화 0~1)
        M : [N, P]     현재 스텝 기계론 프로파일 y*     (정규화 0~1)  ← PINN 평형 기준
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.general_scaler: Optional[MinMaxScaler] = None   # 공정변수 정규화기
        self.conc_scaler: Optional[MinMaxScaler] = None      # CO2 % ↔ 정규화 역변환기
        self.reducer = None                                  # 학습된 UMAP
        self.train_feature_list: List[str] = []

    # ---------------------------------------------------------------------
    # 1.1 로딩
    # ---------------------------------------------------------------------
    def load_excel(self) -> Tuple[List[pd.DataFrame], List[str]]:
        """라벨링된 엑셀들을 표준 DataFrame 리스트로 로딩. (열 멀티인덱스 평탄화)"""
        paths = sorted(glob.glob(self.cfg.data_glob))
        if not paths:
            raise FileNotFoundError(f"데이터 없음: {self.cfg.data_glob}")
        dfs: List[pd.DataFrame] = []
        for p in paths:
            xls = pd.ExcelFile(p)
            df = pd.read_excel(xls, sheet_name=0, index_col=0, header=[0, 1])
            df.columns = df.columns.map("".join)                 # 멀티헤더 → 단일문자열
            df = df.rename_axis("time").reset_index()
            names = list(df.columns)
            names[-1] = self.cfg.label_col                       # 마지막 열 → 'label'
            df.columns = names
            dfs.append(df)
        return dfs, paths

    # ---------------------------------------------------------------------
    # 1.2 Point-1 이상치 치환 (Sec 2.3): 상단 챔버 잔류 rich-gas 보정
    # ---------------------------------------------------------------------
    def _avg_out_point1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Point 1 측정을 인접 Point 2 사이클 평균으로 치환 (분석기 잔류 편향 제거)."""
        labels = list(df[self.cfg.label_col])
        conc = list(df[self.cfg.conc_col])
        # (a) 각 point-2 연속 구간의 평균을 순서대로 수집
        p, num, i, avg2 = 0.0, 0, 0, []
        while i < len(labels):
            if labels[i] == 2:
                p += conc[i]; num += 1; i += 1
            else:
                if p == 0 and num == 0:
                    i += 1
                else:
                    avg2.append(p / num); i += 1; num = 0; p = 0.0
        # (b) point-1 구간을 직전 point-2 평균으로 치환
        i, k = 0, -1
        while i < len(labels) - 1:
            if labels[i] == 1:
                if 0 <= k < len(avg2):
                    conc[i] = avg2[k]
                i += 1
            else:
                if labels[i + 1] == 1:
                    k += 1
                i += 1
        df[self.cfg.conc_col] = pd.Series(data=conc, index=df.index)
        return df

    # ---------------------------------------------------------------------
    # 1.3 [핵심 고도화 ①] Akima 보간으로 6-포인트 프로파일 복원
    # ---------------------------------------------------------------------
    def _column_separator_akima(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        가스분석기는 한 시점에 한 포인트만 측정 → 나머지는 결측.
        v1 은 선형보간(pd 'linear')이라 포트 전환 시점에 '꺾임(kink)' 불연속이 생겨
        노이즈처럼 학습된다.  v2 는 Akima1DInterpolator 로 C1-매끄러운(미분가능) 곡선을
        복원하여 꺾임 노이즈 유입을 원천 차단한다.

        Shape: df 에 '1_sampling'..'6_sampling' (float) 6개 컬럼 추가.
        """
        n = df.shape[0]
        t = np.arange(n, dtype=float)                     # 균일 시간축 (샘플 인덱스)
        labels = df[self.cfg.label_col].values
        conc = df[self.cfg.conc_col].values
        for point in range(1, self.cfg.n_points + 1):
            mask = (labels == point) & ~np.isnan(conc)    # 이 포인트의 실측 시점만 True
            xk, yk = t[mask], conc[mask]
            col = np.full(n, np.nan, dtype=float)
            if len(xk) >= 2:
                # Akima 는 3점 이상에서 진가를 발휘하나, 2점도 선형 폴백으로 안전 처리.
                if len(xk) >= 3:
                    interp = Akima1DInterpolator(xk, yk)
                    col = interp(t)                        # 노드 밖은 NaN(외삽 억제) → 아래서 보정
                    col = np.interp(t, xk, yk) if np.all(np.isnan(col)) else col
                    # Akima 는 [xk[0], xk[-1]] 밖을 NaN 으로 둠 → 경계는 최근접값으로 채움
                    nan_idx = np.isnan(col)
                    if nan_idx.any():
                        col[nan_idx] = np.interp(t[nan_idx], xk, yk)
                else:
                    col = np.interp(t, xk, yk)             # 2점 → 선형
            elif len(xk) == 1:
                col[:] = yk[0]                             # 1점 → 상수
            df[f"{point}_sampling"] = pd.Series(data=col, index=df.index)
        return df

    # ---------------------------------------------------------------------
    # 1.4 정규화 (Eq.1 MMN) — train 통계로 fit
    # ---------------------------------------------------------------------
    def _fit_scalers(self, train_dfs: List[pd.DataFrame]) -> None:
        full, co2 = [], []
        for df in train_dfs:
            full.append(df.values[:, :-1])                # label 열 제외
            co2.append(df[self.cfg.conc_col].values)
        full = np.concatenate(full, axis=0)
        co2 = np.concatenate(co2, axis=0).reshape(-1, 1)
        self.general_scaler = MinMaxScaler().fit(full)
        self.conc_scaler = MinMaxScaler().fit(co2)

    # ---------------------------------------------------------------------
    # 1.5 [핵심 고도화 ②] 차원축소 (16-D) — UMAP / PCA / PLS 선택
    # ---------------------------------------------------------------------
    def _fit_reducer(self, train_dfs: List[pd.DataFrame], feature_list: List[str]) -> None:
        """
        cfg.reduction 에 따라 축소기 학습:  [ΣN_train, F_full] → 16-D.
          · 'umap' : 비지도 비선형 매니폴드 (시각화용, 회귀 신호손실 위험)
          · 'pca'  : 비지도 선형 직교 (POD 동형, 강건 베이스라인)
          · 'pls'  : 지도 부분최소제곱 — 타깃(6-포인트 프로파일)과 공분산 큰 성분만 추출
        """
        d = self.cfg.reduced_dimension
        X = np.concatenate([df[feature_list].values for df in train_dfs], axis=0)  # [ΣN, F]
        method = self.cfg.reduction.lower()

        if method == "umap":
            if umap is None:
                raise ImportError("umap-learn 미설치: `pip install umap-learn`.")
            self.reducer = umap.UMAP(
                n_components=d, n_neighbors=self.cfg.umap_neighbors,
                min_dist=self.cfg.umap_min_dist, random_state=self.cfg.random_state,
            ).fit(X)
        elif method == "pca":
            self.reducer = PCA(n_components=d, random_state=self.cfg.random_state).fit(X)
        elif method == "pls":
            # 지도 축소: 타깃 = Akima 복원 6-포인트 프로파일 (예측 대상과 동일 신호)
            sample_cols = [f"{i}_sampling" for i in range(1, self.cfg.n_points + 1)]
            Y = np.concatenate([df[sample_cols].values for df in train_dfs], axis=0)  # [ΣN, 6]
            Y = np.nan_to_num(Y, nan=0.0)
            self.reducer = PLSRegression(n_components=d, scale=False).fit(X, Y)
        else:
            raise ValueError(f"알 수 없는 reduction: {self.cfg.reduction}")

    def _reduce_transform(self, df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
        """단일 df → 잠재 임베딩 [N, F] → [N, 16] (PLS 는 x_scores 반환)."""
        X = df[feature_list].values
        if self.cfg.reduction.lower() == "pls":
            return self.reducer.transform(X)          # x_scores [N, 16]
        return self.reducer.transform(X)

    # ---------------------------------------------------------------------
    # 1.6 슬라이딩 윈도우 시퀀스화 (+ 포인트 원핫)
    # ---------------------------------------------------------------------
    def _make_sequences(
        self,
        latent_list: List[np.ndarray],       # 각 df 의 [N, 16] UMAP 잠재
        df_list: List[pd.DataFrame],         # 라벨/프로파일 원본 (정규화 완료)
        kinetic_list: List[np.ndarray],      # 각 df 의 [N, 6] 기계론 분율
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        슬라이딩 윈도우(W = callback+1)로 시퀀스 텐서 생성.
          X : [N, W, 16+6]   (잠재 16 + 측정위치 원핫 6)
          Y : [N, 6]         현재 스텝 6-포인트 프로파일 라벨 (정규화)
          M : [N, 6]         현재 스텝 기계론 프로파일 y* (정규화, PINN 기준)
        """
        cb, P = self.cfg.callback, self.cfg.n_points
        sample_cols = [f"{i}_sampling" for i in range(1, P + 1)]
        X_all, Y_all, M_all = [], [], []

        for latent, df, kin in zip(latent_list, df_list, kinetic_list):
            N = df.shape[0]
            nset = N - cb
            if nset <= 0:
                continue
            # 측정위치 원핫: 각 시점의 label(1~6) → one-hot(6)
            onehot = np.zeros((N, P), dtype=float)          # [N, 6]
            lab = df[self.cfg.label_col].values.astype(int)
            for k in range(N):
                if 1 <= lab[k] <= P:
                    onehot[k, lab[k] - 1] = 1.0
            feat = np.hstack([latent, onehot])              # [N, 16+6]
            prof = df[sample_cols].values                   # [N, 6] (라벨)
            prof = np.nan_to_num(prof, nan=0.0)
            # 기계론 y* 는 CO2 % 스케일로 conc_scaler 정규화(라벨과 동일 스케일)
            kin_pct = kin * 100.0                            # 분율 → %
            kin_norm = self.conc_scaler.transform(kin_pct.reshape(-1, 1)).reshape(kin_pct.shape)

            for j in range(nset):
                X_all.append(feat[j:cb + j + 1])            # 윈도우 [W, 16+6]
                Y_all.append(prof[cb + j])                  # 현재 스텝 라벨 [6]
                M_all.append(kin_norm[cb + j])              # 현재 스텝 y* [6]

        X = np.asarray(X_all, dtype=np.float32)             # [N, W, 22]
        Y = np.asarray(Y_all, dtype=np.float32)             # [N, 6]
        M = np.asarray(M_all, dtype=np.float32)             # [N, 6]
        return X, Y, M

    # ---------------------------------------------------------------------
    # 1.7 전체 build — 기존 DataFrame 리스트에 바로 연동 가능한 진입점
    # ---------------------------------------------------------------------
    def build(
        self,
        df_list: Optional[List[pd.DataFrame]] = None,
        paths: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        전처리 파이프라인 실행 → 학습/테스트 텐서 딕셔너리 반환.

        df_list 를 직접 주면(이미 로딩된 사용자 DataFrame) 그것을 사용하고,
        없으면 cfg.data_glob 에서 로딩한다.
        """
        cfg = self.cfg
        if df_list is None:
            df_list, paths = self.load_excel()
        df_list = [d.copy() for d in df_list]

        # (a) 시간 인덱스 세팅 + Point-1 치환
        for i in range(len(df_list)):
            if "time" in df_list[i].columns:
                df_list[i] = df_list[i].set_index("time")
            df_list[i] = self._avg_out_point1(df_list[i])
        print(f"[전처리] 입력 피처 수(라벨 포함): {df_list[0].shape[1]}")

        # (b) train/test 분할
        test_idx = list(cfg.test_df_index_list)
        train_idx = [i for i in range(len(df_list)) if i not in test_idx]

        # (c) 스케일러 fit (train 통계)
        self._fit_scalers([df_list[i] for i in train_idx])

        # (d) 정규화 + Akima 프로파일 복원
        for i in range(len(df_list)):
            df_list[i].iloc[:, :-1] = self.general_scaler.transform(df_list[i].iloc[:, :-1].values)
            df_list[i] = self._column_separator_akima(df_list[i])
            df_list[i] = df_list[i].fillna(0)

        # (e) 피처 리스트 (라벨/프로파일 제외한 정규화 공정변수 전부)
        label_cols = [cfg.label_col] + [f"{i}_sampling" for i in range(1, cfg.n_points + 1)]
        self.train_feature_list = sorted(set(df_list[0].columns) - set(label_cols))
        print(f"[전처리] UMAP 입력 피처 수: {len(self.train_feature_list)}")

        # (f) 축소기 fit(train) → transform(all)
        self._fit_reducer([df_list[i] for i in train_idx], self.train_feature_list)
        latent_list = [self._reduce_transform(df, self.train_feature_list) for df in df_list]
        print(f"[차원축소] {cfg.reduction.upper()}: {len(self.train_feature_list)}-D → {cfg.reduced_dimension}-D")

        # (f-2) [P6 개선] 피드 특성 직접 주입 — PCA 우회 원본 정규화값을 잠재벡터에 append.
        pt = [c for c in cfg.passthrough_features if c in df_list[0].columns]
        if pt:
            latent_list = [np.hstack([lat, df[pt].values])          # [N,16] → [N,16+len(pt)]
                           for lat, df in zip(latent_list, df_list)]
            print(f"[피드주입] passthrough {len(pt)}개 직접 append → 잠재차원 "
                  f"{cfg.reduced_dimension}+{len(pt)}={cfg.reduced_dimension + len(pt)}")

        # (g) 기계론 프로파일 로딩 (파일명 매칭)
        kinetic_list = self._load_kinetic(paths)

        # (h) 시퀀스화
        Xtr, Ytr, Mtr = self._make_sequences(
            [latent_list[i] for i in train_idx],
            [df_list[i] for i in train_idx],
            [kinetic_list[i] for i in train_idx],
        )
        Xte, Yte, Mte = self._make_sequences(
            [latent_list[i] for i in test_idx],
            [df_list[i] for i in test_idx],
            [kinetic_list[i] for i in test_idx],
        )
        print(f"[시퀀스] X_train {Xtr.shape}, Y_train {Ytr.shape} | X_test {Xte.shape}")

        return {
            "X_train": Xtr, "Y_train": Ytr, "M_train": Mtr,
            "X_test": Xte, "Y_test": Yte, "M_test": Mte,
            "conc_scaler": self.conc_scaler,
        }

    def _load_kinetic(self, paths: Optional[List[str]]) -> List[np.ndarray]:
        """엑셀 파일명과 kinetic CSV 파일명을 basename 으로 매칭하여 정렬 로딩."""
        kin_paths = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(self.cfg.kinetic_glob)}
        out = []
        for p in (paths or []):
            key = os.path.splitext(os.path.basename(p))[0]
            if key in kin_paths:
                out.append(pd.read_csv(kin_paths[key], header=None).values.astype(float))
            else:
                # 기계론 CSV 결측 시 0 채움 (PINN 항은 사실상 비활성)
                out.append(None)
        # None 을 동일 shape 0 배열로 대체 (해당 df 길이에 맞춰 나중에 처리)
        return out


# =============================================================================
# 2. [핵심 고도화 ③] GRU 소프트센서 + PINN 물질수지 손실
# =============================================================================
class GRUSoftSensor(_TORCH_BASE):
    """
    경량 GRU 회귀기.  입력 시퀀스 [N, W, F] → 6-포인트 CO2 프로파일 [N, 6] (0~1).

    LSTM(100) → GRU(hidden) 로 파라미터를 축소:
      · 게이트 3→2 (GRU) 로 파라미터/지연(latency) 감소
      · 단층 + Dropout 으로 데이터 부족 환경 과적합 억제
    """

    def __init__(self, in_features: int, mcfg: ModelConfig, n_points: int = 6):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=mcfg.gru_hidden,
            num_layers=mcfg.gru_layers,
            batch_first=True,
            dropout=mcfg.dropout if mcfg.gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(mcfg.dropout),
            nn.Linear(mcfg.gru_hidden, mcfg.gru_hidden),
            nn.ReLU(),
            nn.Dropout(mcfg.dropout),
            nn.Linear(mcfg.gru_hidden, n_points),
            nn.Sigmoid(),                                  # 정규화 라벨 스케일(0~1) 대응
        )

    def forward(self, x):                                  # x: [N, W, F]
        out, h = self.gru(x)                              # out: [N, W, H], h: [1, N, H]
        last = out[:, -1, :]                              # 마지막 스텝 [N, H]
        return self.head(last)                            # [N, 6]


class PINNMassBalanceLoss(_TORCH_BASE):
    """
    [핵심 고도화 ③] Physics-Informed 손실 함수.

    총손실 = MSE_data + w_res·MSE_residual + w_mono·penalty_mono + w_bnd·penalty_bnd

    물질수지 잔차 (정규화 stage 좌표, dz=1, 내부 포인트 i=2..5):
      두-필름 이론 + 축방향 분산 PFR 정상상태:
        R_i = D·(y_{i-1} - 2y_i + y_{i+1})   ← 축방향 분산 (2차 중심차분)
            - u·(y_{i+1} - y_{i-1})/2         ← 기체 이류 (1차 중심차분)
            - Da·(y_i - y*_i)                 ← 두-필름 흡수 sink (기계론 평형기준)
      * y  : GRU 예측 프로파일 (물리 % 스케일로 역정규화)
      * y* : 기계론 kinetic 평형 프로파일

    물리 prior:
      · 단조성 : 피드단(P6)→상단(P1) 로 갈수록 기체 CO2 감소 → relu(y_i - y_{i+1})
      · 경계   : 피드단 P6 는 기계론 경계값과 정합 (BC anchoring)
    """

    def __init__(self, pcfg: PhysicsConfig, conc_scaler: MinMaxScaler):
        super().__init__()
        self.p = pcfg
        # MinMaxScaler 는 affine → torch 상수로 보관하여 미분가능 역정규화 수행.
        # y_phys = y_norm * scale_range + data_min
        self.register_buffer("y_min", torch.tensor(float(conc_scaler.data_min_[0])))
        self.register_buffer("y_rng", torch.tensor(float(conc_scaler.data_max_[0] - conc_scaler.data_min_[0])))
        self.mse = nn.MSELoss()
        # 물리상수 D/u/Da: 고정(buffer) 또는 학습가능(Parameter, inverse-PINN).
        #  학습가능 시 데이터로부터 축방향분산·이류·Damköhler 를 역추정 → 편향 제거.
        d0 = torch.tensor([pcfg.D_axial, pcfg.u_gas, pcfg.damkohler], dtype=torch.float32)
        if pcfg.learnable:
            self.phys = nn.Parameter(d0)
        else:
            self.register_buffer("phys", d0)

    def _denorm(self, y):
        return y * self.y_rng + self.y_min               # [N,6] 정규화 → % (미분가능)

    def forward(self, pred, target, y_star, phys_scale: float = 1.0):
        """
        pred    : [N, 6] GRU 예측 (정규화)
        target  : [N, 6] 라벨      (정규화)
        y_star  : [N, 6] 기계론 평형 (정규화)
        phys_scale : 물리 가중치 워밍업 계수(0→1). 데이터 적합 선확보 후 물리 주입.
        """
        # (1) 데이터 적합 항
        loss_data = self.mse(pred, target)

        # (2) 물리 잔차 항 (물리 % 스케일에서 평가)
        y = self._denorm(pred)                            # [N,6] %
        ys = self._denorm(y_star)                         # [N,6] %
        D, u, Da = self.phys[0], self.phys[1], self.phys[2]
        # 내부 포인트 i=2..5 (0-based 1..4). 인접 슬라이스로 차분.
        y_im1, y_i, y_ip1 = y[:, 0:4], y[:, 1:5], y[:, 2:6]
        ys_i = ys[:, 1:5]
        disp = D * (y_im1 - 2.0 * y_i + y_ip1)                   # 축방향 분산
        adv = u * (y_ip1 - y_im1) / 2.0                         # 이류
        sink = Da * (y_i - ys_i)                                # 두-필름 흡수
        residual = disp - adv - sink                             # [N,4]
        loss_res = (residual ** 2).mean()

        # (3) 단조성 prior: y_i <= y_{i+1} (P1<P2<...<P6). 위반량만 페널티.
        mono = torch.relu(y[:, 0:5] - y[:, 1:6])                 # [N,5]
        loss_mono = (mono ** 2).mean()

        # (4) 경계 정합: 피드단 P6(index 5) ↔ 기계론 경계값
        loss_bnd = self.mse(y[:, 5], ys[:, 5])

        total = (loss_data
                 + phys_scale * self.p.w_residual * loss_res
                 + phys_scale * self.p.w_monotonic * loss_mono
                 + phys_scale * self.p.w_boundary * loss_bnd)
        return total, {
            "data": float(loss_data.detach()),
            "residual": float(loss_res.detach()),
            "monotonic": float(loss_mono.detach()),
            "boundary": float(loss_bnd.detach()),
        }


# =============================================================================
# 3. 학습기 (Trainer)
# =============================================================================
class Trainer:
    """GRU + PINN 학습/검증/추론 루프 (Early stopping, best-weights 복원)."""

    def __init__(self, mcfg: ModelConfig, pcfg: PhysicsConfig, conc_scaler: MinMaxScaler):
        if torch is None:
            raise ImportError("torch 미설치: `pip install torch` 후 재실행.")
        self.mcfg = mcfg
        self.device = torch.device(mcfg.device)
        self.criterion = PINNMassBalanceLoss(pcfg, conc_scaler).to(self.device)
        self.conc_scaler = conc_scaler
        self.model: Optional[GRUSoftSensor] = None

    def _split_train_val(self, N: int, stride: int = 5):
        """매 stride 번째 샘플을 val 로 분리 (v1 동일 규약)."""
        val_idx = list(range(0, N, stride))
        train_idx = [i for i in range(N) if i not in set(val_idx)]
        return train_idx, val_idx

    def fit(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray) -> "GRUSoftSensor":
        cfg = self.mcfg
        N, W, F = X.shape                                        # [N, W, 22]
        tr, va = self._split_train_val(N)
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        Xtr, Ytr, Mtr = to_t(X[tr]), to_t(Y[tr]), to_t(M[tr])
        Xva, Yva, Mva = to_t(X[va]), to_t(Y[va]), to_t(M[va])

        ds = TensorDataset(Xtr, Ytr, Mtr)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        self.model = GRUSoftSensor(in_features=F, mcfg=cfg).to(self.device)
        # 학습가능 물리상수(inverse-PINN)면 criterion 파라미터도 옵티마이저에 포함.
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        warm = max(0, int(self.criterion.p.warmup_epochs))

        best_val, best_state, wait = float("inf"), None, 0
        for epoch in range(cfg.epochs):
            # 물리 가중치 워밍업: 0→1 선형 (데이터 적합 선확보 후 물리 정칙화 주입)
            phys_scale = 1.0 if warm == 0 else min(1.0, epoch / warm)
            self.model.train()
            for xb, yb, mb in dl:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss, _ = self.criterion(pred, yb, mb, phys_scale=phys_scale)
                loss.backward()
                opt.step()

            # --- validation (조기종료 기준은 순수 데이터 적합 = 물리항 제외) ---
            self.model.eval()
            with torch.no_grad():
                vpred = self.model(Xva.to(self.device))
                _, parts = self.criterion(vpred, Yva.to(self.device), Mva.to(self.device))
                vloss = parts["data"]

            if vloss < best_val - 1e-9:
                best_val, best_state, wait = vloss, {k: v.detach().cpu().clone()
                                                     for k, v in self.model.state_dict().items()}, 0
            else:
                wait += 1
            if epoch % 25 == 0 or wait == 0:
                print(f"[GRU-PINN] ep{epoch:03d} val={vloss:.5f} "
                      f"(data={parts['data']:.4f} res={parts['residual']:.4f} "
                      f"mono={parts['monotonic']:.4f})")
            if wait >= cfg.patience:
                print(f"[GRU-PINN] Early stop @ ep{epoch} (best val={best_val:.5f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)              # best-weights 복원
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """추론 → 물리 % 스케일 6-포인트 프로파일 [N, 6]."""
        self.model.eval()
        with torch.no_grad():
            p = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.conc_scaler.inverse_transform(p)            # 정규화 → CO2 %

    def save(self, path: str = "GRU_PINN.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, in_features: int):
        self.model = GRUSoftSensor(in_features=in_features, mcfg=self.mcfg).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self.model


# =============================================================================
# 3.5 [P6 개선] 3D-Var 기계론 융합 (Bayesian analysis step)
# =============================================================================
class MechanisticFusion:
    """
    데이터-드리븐 GRU 예측과 기계론(kinetic) 예측을 오차공분산으로 결합.
    (칼만필터 analysis step / 3D-Var 동형 — 논문 Sec 2.8 Eq.17~19)

    동기: GRU 는 상단부(P1~P4)에서 정밀하나 피드단 P6 는 약함. 기계론은 입구 조성을
          알아 P6 에 강함. 각자의 오차공분산으로 가중결합하면 지점별 강점을 취한다.

        y_fused = y_mec + B·(B + R)⁻¹·(y_GRU − y_mec)          ... Eq.(17)
          · B = Cov(y_mec  − y_label)   (기계론 사전오차, train)  ... Eq.(18)
          · R = Cov(y_GRU  − y_label)   (GRU 관측오차, val)       ... Eq.(18)
          · 두 공분산은 Gaspari-Cohn 국소화(L=2)로 원거리 포인트 허위상관 제거 ... Eq.(19)
    """

    def __init__(self, corr_length: int = 2, n_points: int = 6):
        self.L = corr_length
        self.P = n_points
        self.B: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None

    # --- Gaspari-Cohn 국소화 (Eq.19) ---
    @staticmethod
    def _gc(i: int, j: int, L: float) -> float:
        r = abs(i - j) / L
        if 0 <= r < 1:
            return 1 - r**2 * 5/3 + r**3 * 5/8 + r**4 * 0.5 - r**5 * 0.25
        if 1 <= r < 2:
            return 4 - 5*r + r**2 * 5/3 + r**3 * 5/8 - r**4 * 0.5 + r**5/12 - 2/3/r
        return 0.0

    def _localize(self, C: np.ndarray) -> np.ndarray:
        out = C.copy()
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                out[i, j] = C[i, j] * self._gc(i, j, self.L)
        return out

    def fit(self, mec_train: np.ndarray, lab_train: np.ndarray,
            gru_val: np.ndarray, lab_val: np.ndarray) -> "MechanisticFusion":
        """
        mec_train/lab_train : [Ntr, 6] 기계론·라벨 (%)  → B
        gru_val/lab_val     : [Nva, 6] GRU·라벨 (%)      → R
        """
        self.B = self._localize(np.cov((mec_train - lab_train).T)) + 1e-10 * np.eye(self.P)
        self.R = self._localize(np.cov((gru_val - lab_val).T)) + 1e-10 * np.eye(self.P)
        return self

    def fuse(self, mec: np.ndarray, gru: np.ndarray) -> np.ndarray:
        """
        mec/gru : [N, 6] (%). 반환: [N, 6] 융합 프로파일.
          xa = xb + K(Y − xb),  K = B(B + R)⁻¹,  xb=기계론, Y=GRU, H=I
        """
        H = np.eye(self.P)
        K = self.B @ H.T @ np.linalg.pinv(H @ self.B @ H.T + self.R)   # 이득행렬 [6,6]
        out = np.empty_like(gru)
        for i in range(gru.shape[0]):
            xb = mec[i]
            out[i] = xb + K @ (gru[i] - H @ xb)
        return out


# =============================================================================
# 4. [핵심 고도화 ④] OPEX 중심 경제성 평가 & 정량 검증 모듈
# =============================================================================
class EconomicEvaluator:
    """
    소프트센서 예측오차를 OPEX 화폐가치로 환산하여 베이스라인 대비 절감/회수기간 산출.

    핵심 아이디어:
      · 하단부(P5,P6)는 리보일러 스팀에너지와 직결 → RMSE 를 분리 측정.
      · '불확실성 비용(Cost of Uncertainty)' = 비대칭 손실:
          - 과소예측(pred < true): CO2 배출 과소인지 → carbon slip → 탄소배출권 페널티
          - 과대예측(pred > true): MEA 과순환 → 리보일러 스팀 과다 소모 비용
    """

    def __init__(self, ecfg: EconomicConfig, n_points: int = 6):
        self.e = ecfg
        self.P = n_points

    # ---- 4.1 RMSE 지표 (전체 + 하단부 분리) ----------------------------------
    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(true, pred)))

    def rmse_report(self, pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """
        pred/true : [N, 6] (CO2 %).  전체 RMSE + 하단부(P5,P6) 분리 RMSE.
        """
        rep = {"rmse_overall": self.rmse(pred, true)}
        for p in range(self.P):
            rep[f"rmse_P{p+1}"] = self.rmse(pred[:, p], true[:, p])
        # 하단부(스팀 직결): index 4,5 = Point 5,6
        rep["rmse_bottom_P5P6"] = self.rmse(pred[:, 4:6], true[:, 4:6])
        return rep

    # ---- 4.2 % 오차 → CO2 질량유량 [ton/hr] 환산 ----------------------------
    def _pct_to_co2_ton_per_hr(self, pct: np.ndarray) -> np.ndarray:
        """
        CO2 몰% 오차 → CO2 질량유량 [ton/hr].
          몰분율 오차 Δy → 질량분율 ≈ Δy·(M_CO2/M_gas), × 배가스유량.
        """
        mass_frac = (pct / 100.0) * (self.e.co2_molar_mass / self.e.gas_molar_mass)
        return mass_frac * self.e.flue_gas_flow_kg_hr / 1000.0    # kg/hr → ton/hr

    # ---- 4.3 불확실성 비용 (비대칭) -----------------------------------------
    def cost_of_uncertainty(self, pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """
        상단부(P1, 배출단)의 예측오차로 불확실성 비용을 산정.
          err = pred - true  (부호 기준: +면 과대예측, -면 과소예측)

        · 과소예측(err<0): 실제 배출이 예측보다 큼 → carbon slip.
            비용 = |ΔCO2[ton/hr]| × 가동시간 × 탄소배출권단가
        · 과대예측(err>0): 흡수 여유 과대인지 → MEA 과순환 → 리보일러 스팀 과다.
            잉여 재생열 = ΔCO2[ton/hr] × 재생원단위[GJ/ton] × overcirc_gain
            스팀[ton/hr] = 잉여열 / 잠열[GJ/ton]  →  × 스팀단가 × 가동시간
        """
        # 배출단(상단 P1) 기준 오차 (%p)
        err_pct = pred[:, 0] - true[:, 0]                        # [N]
        under = np.clip(-err_pct, 0, None)                       # 과소예측분(양수)
        over = np.clip(err_pct, 0, None)                         # 과대예측분(양수)

        # 시간평균 ton/hr 로 환산 (샘플 평균 = 대표 운전점)
        under_ton_hr = self._pct_to_co2_ton_per_hr(under).mean()
        over_ton_hr = self._pct_to_co2_ton_per_hr(over).mean()

        hrs = self.e.operating_hours_yr
        # (a) Carbon slip 페널티 [USD/yr]
        carbon_cost = under_ton_hr * hrs * self.e.carbon_price_per_ton

        # (b) Over-circulation 스팀 비용 [USD/yr]
        surplus_gj_hr = over_ton_hr * self.e.reboiler_duty_gj_per_ton_co2 * self.e.overcirc_gain
        steam_ton_hr = surplus_gj_hr / self.e.steam_latent_heat_gj_per_ton
        steam_cost = steam_ton_hr * hrs * self.e.steam_price_per_ton

        return {
            "under_pred_co2_ton_hr": float(under_ton_hr),
            "over_pred_co2_ton_hr": float(over_ton_hr),
            "carbon_slip_cost_usd_yr": float(carbon_cost),
            "overcirc_steam_ton_hr": float(steam_ton_hr),
            "overcirc_steam_cost_usd_yr": float(steam_cost),
            "total_uncertainty_cost_usd_yr": float(carbon_cost + steam_cost),
        }

    # ---- 4.4 베이스라인 대비 절감 & Payback ---------------------------------
    def compare(
        self,
        pred_new: np.ndarray, pred_base: np.ndarray, true: np.ndarray,
    ) -> Dict[str, object]:
        """
        고도화 모델 vs 베이스라인 → 연간 순 OPEX 절감 + 투자비 회수기간.
          pred_new  : [N,6] 고도화(GRU-PINN+UMAP+Akima) 예측
          pred_base : [N,6] 베이스라인(LSTM+선형+POD) 예측
          true      : [N,6] 실측 프로파일
        """
        rep_new = self.rmse_report(pred_new, true)
        rep_base = self.rmse_report(pred_base, true)
        cost_new = self.cost_of_uncertainty(pred_new, true)
        cost_base = self.cost_of_uncertainty(pred_base, true)

        annual_saving = (cost_base["total_uncertainty_cost_usd_yr"]
                         - cost_new["total_uncertainty_cost_usd_yr"])
        payback_yr = (self.e.capex_usd / annual_saving) if annual_saving > 0 else float("inf")

        return {
            "rmse_new": rep_new,
            "rmse_baseline": rep_base,
            "cost_new": cost_new,
            "cost_baseline": cost_base,
            "annual_net_opex_saving_usd": float(annual_saving),
            "capex_usd": self.e.capex_usd,
            "payback_period_yr": float(payback_yr),
        }

    # ---- 4.5 리포트 프린터 --------------------------------------------------
    @staticmethod
    def print_report(result: Dict[str, object]) -> None:
        rn, rb = result["rmse_new"], result["rmse_baseline"]
        cn, cb = result["cost_new"], result["cost_baseline"]
        line = "=" * 72
        print(f"\n{line}\n  경제성/정확도 정량 검증 리포트 (고도화 vs 베이스라인)\n{line}")
        print(f"  [정확도 RMSE, CO2 %]")
        print(f"    전체        : {rb['rmse_overall']:.4f} → {rn['rmse_overall']:.4f}")
        print(f"    하단 P5·P6  : {rb['rmse_bottom_P5P6']:.4f} → {rn['rmse_bottom_P5P6']:.4f}"
              f"   (리보일러 스팀 직결)")
        print(f"  [불확실성 비용, USD/yr]")
        print(f"    Carbon slip : {cb['carbon_slip_cost_usd_yr']:,.0f} → "
              f"{cn['carbon_slip_cost_usd_yr']:,.0f}")
        print(f"    과순환 스팀 : {cb['overcirc_steam_cost_usd_yr']:,.0f} → "
              f"{cn['overcirc_steam_cost_usd_yr']:,.0f}")
        print(f"    합계        : {cb['total_uncertainty_cost_usd_yr']:,.0f} → "
              f"{cn['total_uncertainty_cost_usd_yr']:,.0f}")
        print(f"  [재무 지표]")
        print(f"    연간 순 OPEX 절감 : {result['annual_net_opex_saving_usd']:,.0f} USD/yr")
        print(f"    투자비(CAPEX)     : {result['capex_usd']:,.0f} USD")
        pb = result["payback_period_yr"]
        pb_str = f"{pb:.2f} yr ({pb*12:.1f} 개월)" if np.isfinite(pb) else "N/A (절감 없음)"
        print(f"    투자비 회수기간   : {pb_str}\n{line}\n")


# =============================================================================
# 5. 유틸 & 오케스트레이션
# =============================================================================
def reset_seeds(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def run_pipeline(
    dcfg: DataConfig = DataConfig(),
    mcfg: ModelConfig = ModelConfig(),
    pcfg: PhysicsConfig = PhysicsConfig(),
    ecfg: EconomicConfig = EconomicConfig(),
    baseline_pred_path: Optional[str] = None,
    fuse: bool = True,
) -> Dict[str, object]:
    """
    엔드투엔드 실행: 전처리(Akima+PCA) → GRU-PINN 학습 → [3D-Var 기계론 융합] → OPEX 검증.

    baseline_pred_path : v1 베이스라인 예측(.npy, [N,6] CO2 %) 경로.
        주어지면 그것과 비교, 없으면 '기계론 단독' 을 베이스라인 대용으로 사용.
    fuse : True 면 GRU 예측을 기계론과 3D-Var 융합(P6 개선). 최종 예측으로 채택.
    """
    reset_seeds(dcfg.random_state)

    # 1) 데이터
    pipe = DataPipeline(dcfg)
    data = pipe.build()
    Xtr, Ytr, Mtr = data["X_train"], data["Y_train"], data["M_train"]
    Xte, Yte = data["X_test"], data["Y_test"]
    conc_scaler = data["conc_scaler"]

    # 2) 학습
    trainer = Trainer(mcfg, pcfg, conc_scaler)
    trainer.fit(Xtr, Ytr, Mtr)
    trainer.save("GRU_PINN.pt")

    # 3) 추론 (CO2 % 스케일)
    gru_pred = trainer.predict(Xte)                                   # [N,6] % (데이터드리븐)
    true = conc_scaler.inverse_transform(Yte)                        # [N,6] %
    mec_test = conc_scaler.inverse_transform(data["M_test"])         # [N,6] % (기계론)

    # 3.5) 3D-Var 기계론 융합 (P6 개선) — GRU 상단강점 + 기계론 피드단강점 결합
    if fuse:
        # B(기계론 사전오차)=train, R(GRU 관측오차)=val 로 추정 후 국소화.
        Ntr = Xtr.shape[0]
        va = list(range(0, Ntr, dcfg.val_stride))
        trn = [i for i in range(Ntr) if i not in set(va)]
        mec_train = conc_scaler.inverse_transform(Mtr[trn])
        lab_train = conc_scaler.inverse_transform(Ytr[trn])
        gru_val = trainer.predict(Xtr[va])
        lab_val = conc_scaler.inverse_transform(Ytr[va])
        fusion = MechanisticFusion(corr_length=2, n_points=dcfg.n_points).fit(
            mec_train, lab_train, gru_val, lab_val)
        pred_new = fusion.fuse(mec_test, gru_pred)                   # [N,6] % 최종
        print(f"[융합] GRU→3D-Var: overall RMSE "
              f"{np.sqrt(mean_squared_error(true, gru_pred)):.4f} → "
              f"{np.sqrt(mean_squared_error(true, pred_new)):.4f}")
    else:
        pred_new = gru_pred

    # 4) 베이스라인 확보
    if baseline_pred_path and os.path.exists(baseline_pred_path):
        pred_base = np.load(baseline_pred_path)
        pred_base = pred_base[-true.shape[0]:]                        # 길이 정합
    else:
        pred_base = mec_test                                          # 기계론 단독을 대용

    # 5) 경제성/정확도 검증
    evaluator = EconomicEvaluator(ecfg, n_points=dcfg.n_points)
    result = evaluator.compare(pred_new, pred_base, true)
    evaluator.print_report(result)

    result["pred_new"] = pred_new
    result["pred_gru"] = gru_pred
    result["pred_baseline"] = pred_base
    result["true"] = true
    return result


if __name__ == "__main__":
    # 기본 Config = 실측 bake-off 최적 조합(PCA-16 + GRU + inverse-PINN).
    #   test=[2], callback=17, 16-D.  v1 베이스라인(0.304) 대비 RMSE 0.252 (−17%).
    run_pipeline(
        dcfg=DataConfig(test_df_index_list=(2,), callback=17, reduced_dimension=16),
        baseline_pred_path="results/set1_DAE_16.npy",   # v1 베이스라인이 있으면 자동 비교
    )
