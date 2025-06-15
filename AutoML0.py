"""
AutoML with LightGBM – 未来予測モード付き（コメント & Docstring 完全版）
========================================================================

概要
----
アップロードされた CSV を用い、LightGBM による自動モデリングを行う
Streamlit アプリです。下記の機能を備えています。

1. **通常モード** : 回帰または分類モデルを自動構築し評価
2. **時系列モード** : 行番号で *n* 行先の目的変数を予測（未来予測）
3. **RandomizedSearchCV** : 最大 30 試行でハイパーパラ探索
4. **SHAP** : ボタン押下でモデル解釈を実行
5. **AI 解説** : SHAP 結果を含むビジネス向けレポートを
   OpenAI API でストリーミング生成
6. **モデル / ハイパラ ダウンロード** : pkl / json 形式で保存可能
"""

from __future__ import annotations

import io
import json
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import randint, uniform
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# -- OpenAI 初期化 ----------------------------------------------------------
try:
    from openai import OpenAI  # SDK ≥ 1.0

    _OPENAI_OK = True
except ImportError:  # OpenAI SDK が無い場合でもアプリは動作可能
    _OPENAI_OK = False


# ==========================================================================
#                               UTILITIES
# ==========================================================================
def detect_encoding(b: bytes) -> Optional[str]:
    """
    バイト列の先頭 10 KB からファイルエンコーディングを推定する。

    1. `charset_normalizer` がインストールされていればそちらを優先  
    2. 無ければ `chardet` をフォールバックとして利用

    Parameters
    ----------
    b : bytes
        読み込んだファイルのバイト列

    Returns
    -------
    str or None
        推定したエンコーディング名。判定不能の場合は `None`
    """
    try:
        # Python 3.11 以降では標準添付
        from charset_normalizer import from_bytes

        best = from_bytes(b[:10_000]).best()
        return best.encoding if best else None
    except ImportError:
        import chardet

        res = chardet.detect(b[:10_000])
        return res["encoding"] or None


def load_data(uploaded_file, enc_manual: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Streamlit FileUploader で受け取った CSV を読み込み、
    最大 5,000 行に制限して返す。

    Parameters
    ----------
    uploaded_file : UploadedFile
        `st.file_uploader` が返すオブジェクト
    enc_manual : str, optional
        ユーザーが明示的に選択したエンコーディング

    Returns
    -------
    pd.DataFrame or None
        読み込み成功時は DataFrame、失敗した場合は None
    """
    if uploaded_file is None:
        return None

    try:
        b = uploaded_file.getvalue()
        enc = enc_manual or detect_encoding(b) or "utf-8"
        # nrows でメモリ使用量を抑制
        df = pd.read_csv(io.BytesIO(b), encoding=enc, nrows=5_000)
        if len(df) == 5_000:
            st.info("5,000 行を超えたため先頭 5,000 行のみ読み込みました。")
        return df
    except Exception as e:  # 例外の詳細はユーザに表示
        st.error(f"CSV 読み込みでエラーが発生しました: {e}")
        return None


def determine_task(df: pd.DataFrame, target: str) -> str:
    """
    目的変数のデータ型から回帰 or 分類をざっくり判定。

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    target : str
        目的変数列名

    Returns
    -------
    str
        `"regression"` または `"classification"`
    """
    return "regression" if pd.api.types.is_numeric_dtype(df[target]) else "classification"


def preprocess(
    df: pd.DataFrame, target: str, task: str
) -> Tuple[pd.DataFrame, np.ndarray, Optional[LabelEncoder]]:
    """
    LightGBM 用の前処理を行う。

    - 目的変数欠損行を削除
    - `object` 列を `category` に変換
    - 分類タスク時は `LabelEncoder` で数値化

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    target : str
        目的変数列
    task : str
        `"regression"` or `"classification"`

    Returns
    -------
    X : pd.DataFrame
        特徴量データ
    y : np.ndarray
        目的変数配列
    le : LabelEncoder or None
        分類タスク時はエンコーダ、回帰なら None
    """
    df = df.dropna(subset=[target]).copy()
    X = df.drop(columns=[target])
    y = df[target].values

    le: Optional[LabelEncoder] = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # object → category 変換
    for c in X.columns:
        if X[c].dtype == "object" or pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X, y, le


def align_categories(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    カテゴリ列のカテゴリ集合を train / test で一致させ、
    未知カテゴリによる推論エラーを防止する。

    Parameters
    ----------
    train, test : pd.DataFrame
        学習・評価用データフレーム
    """
    for c in train.select_dtypes(include=["category"]).columns:
        cats = train[c].cat.categories.union(test[c].cat.categories)
        train[c] = train[c].cat.set_categories(cats)
        test[c] = test[c].cat.set_categories(cats)


# ==========================================================================
#                   時系列用ユーティリティ
# ==========================================================================
def create_future_target(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """
    目的変数を *horizon* 行先にシフトし、未来予測用データを生成。

    末尾 `horizon` 行はターゲットが NaN になるため除外する。

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    target : str
        予測対象列
    horizon : int
        どれだけ先を予測するか（行数）

    Returns
    -------
    pd.DataFrame
        シフト済みデータ
    """
    df = df.copy()
    df[target] = df[target].shift(-horizon)  # 未来へシフト
    return df.dropna(subset=[target]).reset_index(drop=True)


# ==========================================================================
#              RandomizedSearchCV & モデル関連
# ==========================================================================
def param_distributions(task: str) -> Dict:
    """
    RandomizedSearchCV で使用するパラメータ分布を返す。

    Returns
    -------
    dict
        パラメータ名 → scipy.stats 分布 or list
    """
    dist: Dict = {
        "n_estimators": randint(100, 601),       # 100〜600
        "learning_rate": uniform(0.01, 0.19),    # 0.01〜0.2
        "num_leaves": randint(15, 129),          # 15〜128
        "max_depth": randint(4, 13),             # 4〜12
    }
    dist["objective"] = (
        ["regression", "regression_l1"]
        if task == "regression"
        else ["binary", "multiclass"]
    )
    return dist


def create_lgbm(task: str):
    """
    タスク種別に応じた LightGBM モデルインスタンスを返す。
    """
    base = dict(random_state=42)
    return lgb.LGBMRegressor(**base) if task == "regression" else lgb.LGBMClassifier(**base)


@st.cache_resource(show_spinner=False, ttl=3_600)
def train_random_search(X: pd.DataFrame, y: np.ndarray, task: str):
    """
    RandomizedSearchCV によるハイパーパラ探索＋学習を実施。

    Parameters
    ----------
    X, y : 入力特徴量・目的変数
    task : str
        'regression' or 'classification'

    Returns
    -------
    best_estimator_, best_params_
    """
    rs = RandomizedSearchCV(
        estimator=create_lgbm(task),
        param_distributions=param_distributions(task),
        n_iter=30,  # 試行回数
        scoring="neg_root_mean_squared_error" if task == "regression" else "f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )
    rs.fit(X, y, categorical_feature="auto")
    return rs.best_estimator_, rs.best_params_


# ==========================================================================
#                       評価指標と可視化
# ==========================================================================
def evaluate(model, X_test, y_test, task: str, le: Optional[LabelEncoder]) -> Dict:
    """
    学習済みモデルを評価し、指標を辞書で返す。
    """
    y_pred = model.predict(X_test)
    res: Dict[str, float | np.ndarray | str] = {}
    if task == "regression":
        res["R²"] = r2_score(y_test, y_pred)
        res["RMSE"] = mean_squared_error(y_test, y_pred, squared=False)
    else:
        res["Accuracy"] = accuracy_score(y_test, y_pred)
        avg = "macro" if len(set(y_test)) > 2 else "binary"
        res["F1"] = f1_score(y_test, y_pred, average=avg)
        res["Confusion"] = confusion_matrix(y_test, y_pred)
        res["Report"] = classification_report(
            y_test, y_pred, target_names=le.classes_, digits=3, zero_division=0
        )
    return res


def plot_importance(model, cols):
    """
    特徴量重要度上位 20 件を水平バーで可視化。
    """
    imp = (
        pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    return px.bar(imp, x="importance", y="feature", orientation="h", title="特徴量重要度 (Top 20)")


def plot_reg(y_test, y_pred):
    """
    回帰タスクの実測 vs 予測散布図。3 000 点にサンプリングして描画負荷を低減。
    """
    idx = np.random.choice(len(y_test), min(3_000, len(y_test)), replace=False)
    fig = px.scatter(
        x=y_test[idx],
        y=y_pred[idx],
        labels={"x": "実測値", "y": "予測値"},
        title="実測 vs 予測 (サンプル)",
        trendline="ols",
        trendline_color_override="red",
    )
    fig.add_shape(
        type="line",
        x0=y_test.min(),
        y0=y_test.min(),
        x1=y_test.max(),
        y1=y_test.max(),
        line=dict(dash="dash"),
    )
    return fig


def plot_cm(cm, labels):
    """
    混同行列をヒートマップで表示。
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="予測ラベル", y="実測ラベル"),
        x=labels,
        y=labels,
        title="混同行列",
    )
    fig.update_xaxes(side="top")
    return fig


# ==========================================================================
#                   AI Explanation (stream)
# ==========================================================================
def stream_explanation(metrics: Dict, shap_top: pd.DataFrame, task: str, api_key: str):
    """
    OpenAI Chat Completion をストリーミングしながら、
    活動指標 & SHAP 重要度に基づくレポートを生成・表示する。
    """
    if not _OPENAI_OK:
        st.error("openai パッケージがインストールされていません。")
        return

    client = OpenAI(api_key=api_key)

    prompt = f"""
あなたは熟練データサイエンティストです。以下の情報を基に、
専門用語を最小限に抑えて非エンジニア向けに分かりやすく解説してください。

### モデル性能
{chr(10).join(f"- {k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float))}

### SHAP 重要特徴量 (Top 5)
{chr(10).join(f"{i+1}. {r.Feature} (平均影響度: {r.MeanAbsSHAP:.4f})" for i, r in shap_top.head(5).iterrows())}

1. 指標に基づく性能評価（良い/悪い と根拠）
2. 特徴量が結果へどう寄与したか
3. 得られた示唆と次の施策
"""

    def gen():
        """
        OpenAI からのストリーミングレスポンスを逐次 yield する
        ジェネレータ。
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはデータ分析の専門家です。"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.4,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    st.subheader("🤖 AI 解説 (SHAP 連動)")
    st.write_stream(gen)


# ==========================================================================
#                          Streamlit APP
# ==========================================================================
def main() -> None:
    """
    Streamlit アプリのエントリポイント。
    """
    st.set_page_config(page_title="AutoML with LightGBM", layout="wide")
    st.title("🤖 AutoML (LightGBM) – 未来予測モード付き")

    # -------- Sidebar -----------------------------------------------------
    with st.sidebar:
        st.header("1️⃣ ファイル")
        file = st.file_uploader("CSV を選択", type="csv")
        enc_opt = st.selectbox("エンコーディング", ["自動判定", "utf-8", "shift_jis", "cp932"], 0)

        st.header("2️⃣ OpenAI API キー (任意)")
        api_key = st.text_input("API キー", type="password")

    # -------- Session State (初期化) --------------------------------------
    for k in ["df_proc", "model", "X_test", "y_test", "task", "le", "metrics"]:
        st.session_state.setdefault(k, None)

    # -------- Data Load & Preprocess --------------------------------------
    if file:
        df_raw = load_data(file, None if enc_opt == "自動判定" else enc_opt)
        if df_raw is not None:
            st.header("🧹 前処理")
            with st.expander("プレビュー", expanded=True):
                st.dataframe(df_raw.head())
                st.text(f"{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

            col1, col2 = st.columns(2)
            with col1:
                skip = st.number_input("削除する先頭行数", 0, 50, 0)
            with col2:
                drops = st.multiselect("削除する列", df_raw.columns)

            if st.button("前処理を適用", type="primary"):
                dfp = df_raw.copy()
                if skip:
                    dfp = dfp.iloc[skip:].reset_index(drop=True)
                if drops:
                    dfp.drop(columns=drops, inplace=True)
                st.session_state.df_proc = dfp
                st.success("前処理完了")

    # -------- Training ----------------------------------------------------
    if st.session_state.df_proc is not None:
        dfp = st.session_state.df_proc
        st.header("🚀 モデル学習")
        st.dataframe(dfp.head())

        target = st.selectbox("目的変数", dfp.columns)

        # ----- 時系列モード設定 -----
        st.subheader("時系列設定 (任意)")
        ts_mode = st.checkbox("将来の値を予測する (時系列モード)")
        horizon = 1
        if ts_mode:
            horizon = st.number_input("何行先を予測？", 1, 1000, 1)
            df_future = create_future_target(dfp, target, horizon)
            st.info(f"{horizon} 行先の {target} を予測対象としました。 (行数: {len(df_future)}/{len(dfp)})")
            df_use = df_future
        else:
            df_use = dfp

        task_auto = determine_task(df_use, target)
        task = st.radio("タスク種別", ["regression", "classification"], index=0 if task_auto == "regression" else 1)

        if st.button("📈 学習開始"):
            try:
                # ---------- 前処理 ----------
                with st.spinner("データ前処理..."):
                    X, y, le = preprocess(df_use, target, task)

                    if ts_mode:
                        # シャッフルせず、前 80% を学習、後 20% をテスト
                        split = int(len(X) * 0.8)
                        X_tr, X_te = X.iloc[:split], X.iloc[split:]
                        y_tr, y_te = y[:split], y[split:]
                    else:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y if task == "classification" else None,
                        )

                    align_categories(X_tr, X_te)

                # ---------- ハイパパラ探索 ----------
                with st.spinner("RandomizedSearch (30 試行)..."):
                    model, best_params = train_random_search(X_tr, y_tr, task)

                # ---------- 評価 ----------
                mets = evaluate(model, X_te, y_te, task, le)

                # 状態保存（SHAP など後工程で使用）
                st.session_state.update(
                    dict(model=model, X_test=X_te, y_test=y_te, task=task, le=le, metrics=mets)
                )

                st.success("学習完了！")

                # ---------- 結果表示 ----------
                st.header("📊 結果")
                lcol, rcol = st.columns(2)

                with lcol:
                    st.subheader("性能指標")
                    for k, v in mets.items():
                        if isinstance(v, float):
                            st.metric(k, f"{v:.4f}")
                    if task == "classification":
                        st.plotly_chart(plot_cm(mets["Confusion"], le.classes_), use_container_width=True)
                        with st.expander("詳細レポート"):
                            st.text(mets["Report"])

                with rcol:
                    st.subheader("特徴量重要度")
                    st.plotly_chart(plot_importance(model, X.columns), use_container_width=True)
                    if task == "regression":
                        st.plotly_chart(plot_reg(y_te, model.predict(X_te)), use_container_width=True)

                # ---------- ダウンロード ----------
                st.download_button("モデル (.pkl)", pickle.dumps(model), "lgbm_model.pkl")
                st.download_button(
                    "ハイパラ (.json)",
                    json.dumps(best_params, ensure_ascii=False, indent=2),
                    "best_params.json",
                )

            except Exception as e:
                st.error("学習中にエラーが発生しました。")
                # 内部ログにのみフルスタックを出力
                with open("error_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: {e}\n")

    # -------- SHAP & AI Explanation --------------------------------------
    if st.session_state.model is not None:
        st.header("🔍 SHAP で詳細解釈")
        if st.button("SHAP を計算し AI 解説"):
            try:
                import shap

                # --- SHAP 計算 ---
                with st.spinner("SHAP 計算中..."):
                    explainer = shap.TreeExplainer(st.session_state.model)
                    shap_vals = explainer.shap_values(st.session_state.X_test, check_additivity=False)

                st.subheader("SHAP Summary Plot")
                shap.summary_plot(shap_vals, st.session_state.X_test, show=False)
                st.pyplot(autoresize=True, clear_figure=True)

                # SHAP 値の平均絶対値 (多クラスはクラス平均)
                shap_abs = (
                    np.mean([np.abs(v) for v in shap_vals], axis=0)
                    if isinstance(shap_vals, list)
                    else np.abs(shap_vals).mean(axis=0)
                )
                shap_df = (
                    pd.DataFrame(
                        {"Feature": st.session_state.X_test.columns, "MeanAbsSHAP": shap_abs}
                    )
                    .sort_values("MeanAbsSHAP", ascending=False)
                    .reset_index(drop=True)
                )

                # --- AI 解説 ---
                if api_key:
                    with st.spinner("AI 解説生成中..."):
                        stream_explanation(st.session_state.metrics, shap_df, st.session_state.task, api_key)
                else:
                    st.info("OpenAI API キー未入力のため AI 解説は生成されません。")

            except ImportError:
                st.error("`shap` パッケージがインストールされていません。`pip install shap` を実行してください。")
            except Exception as e:
                st.error(f"SHAP/解説生成中にエラーが発生しました: {e}")


# ------------------ エントリポイント ------------------
if __name__ == "__main__":
    main()
