# SimpleAutoML-LightGBM-SHAP

SimpleAutoML は、アップロードした CSV ファイルから自動的に LightGBM モデルを学習する軽量な Streamlit アプリケーションです。回帰と分類の両タスクをサポートし、基本的なハイパーパラメータ探索や SHAP を用いたモデル解釈機能を備えています。時系列モードを有効にすると、ターゲット列を数行先まで予測できます。

## 特徴

- CSV をアップロードするだけでタスク（回帰／分類）を自動判別
- ターゲット列をシフトして時系列予測が可能
- `RandomizedSearchCV` によるハイパーパラメータチューニング
- SHAP によるモデル解釈
- OpenAI API を利用した AI 生成の説明（任意）
- 学習済みモデルと最適なハイパーパラメータをダウンロード可能

## 必要なライブラリ

以下のコマンドで依存ライブラリをインストールします。

```bash
pip install -r requirements.txt
```

## 使い方

以下のコマンドで Streamlit アプリを起動します。

```bash
streamlit run AutoML0.py
```

データセットをアップロードし、ターゲット列や各種オプションを指定してください。OpenAI API キーを入力すると、SHAP 出力の簡単な説明を生成できます。

## ライセンス

このプロジェクトは MIT License の下で公開されています。詳細は [LICENSE](LICENSE) を参照してください。
