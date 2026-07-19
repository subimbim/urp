#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
 CO2-MEA 흡수탑 소프트센서 — 데이터 전처리 & Akima 보간 모듈
================================================================================
co2_soft_sensor_v2.py 에서 '전처리 + 결측 보간' 부분만 분리한 독립 모듈.
(차원축소/GRU/PINN/융합/경제성 제외)

파이프라인 순서
--------------------------------------------------------------------------------
  1) 엑셀 로딩          : 멀티헤더 평탄화 + 마지막 열을 'label' 로 표준화
  2) Point-1 이상치 치환 : 상단 챔버 잔류 rich-gas 보정(인접 Point-2 평균 치환)
  3) MinMax 정규화       : train 통계로 fit (Eq.1 MMN)  ─ conc_scaler 로 % 역변환
  4) [핵심] Akima 보간   : 6개 샘플링 포인트 프로파일을 C1-매끄러운 곡선으로 복원
                          → 선형보간의 포트전환 '꺾임(kink) 노이즈'를 원천 차단

Shape 규약:  N=시점 수, P=6 샘플링 포인트
================================================================================
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessConfig:
    """전처리 설정."""
    data_glob: str = "data/withLabel/1*.xlsx"   # 라벨링된 공정 기록(엑셀)
    test_df_index_list: Tuple[int, ...] = (2,)  # 스케일러 fit 에서 제외할 테스트셋 인덱스
    n_points: int = 6                           # 흡수탑 샘플링 포인트 수
    conc_col: str = "AT400(CO2 %)"              # 가스분석기 CO2 농도 컬럼
    label_col: str = "label"                    # 측정 위치(1~6) 라벨 컬럼


class Preprocessor:
    """엑셀 → (Point-1 치환) → (MinMax 정규화) → (Akima 보간) 전처리기."""

    def __init__(self, cfg: PreprocessConfig = PreprocessConfig()):
        self.cfg = cfg
        self.general_scaler: Optional[MinMaxScaler] = None   # 공정변수 정규화기
        self.conc_scaler: Optional[MinMaxScaler] = None      # CO2 % ↔ 정규화 역변환기

    # ------------------------------------------------------------------ #
    # 1) 로딩
    # ------------------------------------------------------------------ #
    def load_excel(self) -> Tuple[List[pd.DataFrame], List[str]]:
        """라벨링된 엑셀들을 표준 DataFrame 리스트로 로딩(멀티헤더 평탄화)."""
        paths = sorted(glob.glob(self.cfg.data_glob))
        if not paths:
            raise FileNotFoundError(f"데이터 없음: {self.cfg.data_glob}")
        dfs: List[pd.DataFrame] = []
        for p in paths:
            xls = pd.ExcelFile(p)
            df = pd.read_excel(xls, sheet_name=0, index_col=0, header=[0, 1])
            df.columns = df.columns.map("".join)             # 멀티헤더 → 단일문자열
            df = df.rename_axis("time").reset_index()
            names = list(df.columns)
            names[-1] = self.cfg.label_col                   # 마지막 열 → 'label'
            df.columns = names
            dfs.append(df)
        return dfs, paths

    # ------------------------------------------------------------------ #
    # 2) Point-1 이상치 치환 (상단 챔버 잔류 rich-gas 보정)
    # ------------------------------------------------------------------ #
    def avg_out_point1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Point 1 측정을 직전 Point 2 사이클 평균으로 치환(분석기 잔류 편향 제거)."""
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

    # ------------------------------------------------------------------ #
    # 3) MinMax 정규화 (train 통계로 fit)
    # ------------------------------------------------------------------ #
    def fit_scalers(self, train_dfs: List[pd.DataFrame]) -> None:
        """train 데이터 통계로 general/conc 스케일러 fit (Eq.1 MMN)."""
        full, co2 = [], []
        for df in train_dfs:
            full.append(df.values[:, :-1])                   # label 열 제외
            co2.append(df[self.cfg.conc_col].values)
        full = np.concatenate(full, axis=0)
        co2 = np.concatenate(co2, axis=0).reshape(-1, 1)
        self.general_scaler = MinMaxScaler().fit(full)
        self.conc_scaler = MinMaxScaler().fit(co2)

    # ------------------------------------------------------------------ #
    # 4) [핵심] Akima 보간으로 6-포인트 프로파일 복원
    # ------------------------------------------------------------------ #
    def column_separator_akima(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        가스분석기는 한 시점에 한 포인트만 측정 → 나머지는 결측.
        선형보간은 포트 전환 시점에 '꺾임(kink)' 불연속이 생겨 노이즈처럼 학습되므로,
        Akima1DInterpolator 로 C1-매끄러운(미분가능) 곡선을 복원한다.

        Shape: df 에 '1_sampling'..'6_sampling' (float) 6개 컬럼 추가.
        """
        n = df.shape[0]
        t = np.arange(n, dtype=float)                        # 균일 시간축(샘플 인덱스)
        labels = df[self.cfg.label_col].values
        conc = df[self.cfg.conc_col].values
        for point in range(1, self.cfg.n_points + 1):
            mask = (labels == point) & ~np.isnan(conc)       # 이 포인트 실측 시점만 True
            xk, yk = t[mask], conc[mask]
            col = np.full(n, np.nan, dtype=float)
            if len(xk) >= 3:
                interp = Akima1DInterpolator(xk, yk)
                col = interp(t)                              # 노드 밖은 NaN(외삽 억제)
                if np.all(np.isnan(col)):
                    col = np.interp(t, xk, yk)
                nan_idx = np.isnan(col)                      # 경계는 최근접값으로 보정
                if nan_idx.any():
                    col[nan_idx] = np.interp(t[nan_idx], xk, yk)
            elif len(xk) == 2:
                col = np.interp(t, xk, yk)                   # 2점 → 선형 폴백
            elif len(xk) == 1:
                col[:] = yk[0]                               # 1점 → 상수
            df[f"{point}_sampling"] = pd.Series(data=col, index=df.index)
        return df

    # ------------------------------------------------------------------ #
    # 전체 파이프라인 진입점
    # ------------------------------------------------------------------ #
    def run(
        self,
        df_list: Optional[List[pd.DataFrame]] = None,
        paths: Optional[List[str]] = None,
    ) -> List[pd.DataFrame]:
        """
        전처리 실행 → 정규화 + Akima 프로파일이 채워진 DataFrame 리스트 반환.
        df_list 를 직접 주면 그것을 사용, 없으면 cfg.data_glob 에서 로딩.
        """
        cfg = self.cfg
        if df_list is None:
            df_list, paths = self.load_excel()
        df_list = [d.copy() for d in df_list]

        # (a) 시간 인덱스 세팅 + Point-1 치환
        for i in range(len(df_list)):
            if "time" in df_list[i].columns:
                df_list[i] = df_list[i].set_index("time")
            df_list[i] = self.avg_out_point1(df_list[i])

        # (b) train 통계로 스케일러 fit
        test_idx = set(cfg.test_df_index_list)
        train_dfs = [df_list[i] for i in range(len(df_list)) if i not in test_idx]
        self.fit_scalers(train_dfs)

        # (c) 정규화 + Akima 보간 (전체 df)
        for i in range(len(df_list)):
            df_list[i].iloc[:, :-1] = self.general_scaler.transform(df_list[i].iloc[:, :-1].values)
            df_list[i] = self.column_separator_akima(df_list[i])
            df_list[i] = df_list[i].fillna(0)

        return df_list


if __name__ == "__main__":
    pre = Preprocessor(PreprocessConfig(test_df_index_list=(2,)))
    dfs = pre.run()
    sample_cols = [f"{i}_sampling" for i in range(1, 7)]
    print(f"[전처리 완료] 데이터셋 {len(dfs)}개")
    print(f"  첫 데이터셋 shape: {dfs[0].shape}")
    prof = dfs[0][sample_cols]
    print(f"  Akima 프로파일 결측(NaN) 수: {int(prof.isna().sum().sum())}")
    print(f"  conc_scaler CO2% 범위: {pre.conc_scaler.data_min_[0]:.3f} ~ {pre.conc_scaler.data_max_[0]:.3f}")
