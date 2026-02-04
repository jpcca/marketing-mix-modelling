# Conjura MMM Dataset

このディレクトリには、eコマースブランドのマーケティング・ミックス・モデリング（MMM）用データセットが含まれています。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `conjura_mmm_data.csv` | メインデータセット（132,760行） |
| `conjura_mmm_data_dictionary.tsv` | フィールド定義辞書 |

## データ構造

### 識別子・メタデータ

| カラム | 型 | 説明 |
|--------|-----|------|
| `mmm_timeseries_id` | string | 時系列の一意識別子 |
| `organisation_id` | string | eコマースブランドの匿名ID |
| `organisation_vertical` | string | 商品カテゴリ（Google eCommerce taxonomy準拠） |
| `organisation_subvertical` | string | 商品サブカテゴリ |
| `organisation_marketing_sources` | string | 使用広告プラットフォーム（Google, Meta, TikTok） |
| `organisation_primary_territory_name` | string | 主要販売地域 |
| `territory_name` | string | 地域（"All Territories" または国別） |
| `date_day` | date | 観測日（各時系列は最低449日連続） |
| `currency_code` | string | 通貨コード |

### ターゲット変数（購入指標）

| カラム | 説明 |
|--------|------|
| `first_purchases` | 新規顧客の購入数（アクイジション） |
| `first_purchases_units` | 新規顧客の購入ユニット数 |
| `first_purchases_original_price` | 新規顧客の割引前売上 |
| `first_purchases_gross_discount` | 新規顧客への割引額 |
| `all_purchases` | 全顧客の購入数 |
| `all_purchases_units` | 全顧客の購入ユニット数 |
| `all_purchases_original_price` | 全顧客の割引前売上 |
| `all_purchases_gross_discount` | 全顧客への割引額 |

**注意**: `gross_discount` は購入完了後のみ取得可能。制御変数として使用する際はデータリーケージに注意。

### 広告費（説明変数）

| プラットフォーム | チャネル | カラム |
|-----------------|---------|--------|
| **Google** | 検索広告（非ブランド） | `google_paid_search_spend` |
| | ショッピング広告 | `google_shopping_spend` |
| | Performance Max | `google_pmax_spend` |
| | ディスプレイ広告 | `google_display_spend` |
| | 動画広告 | `google_video_spend` |
| **Meta** | Facebook広告 | `meta_facebook_spend` |
| | Instagram広告 | `meta_instagram_spend` |
| | その他 | `meta_other_spend` |
| **TikTok** | TikTok広告 | `tiktok_spend` |

### エンゲージメント指標

各有料チャネルに対応：
- `<platform>_<channel>_clicks` - 広告クリック数
- `<platform>_<channel>_impressions` - 広告インプレッション数

### オーガニックトラフィック

| カラム | 説明 |
|--------|------|
| `direct_clicks` | 直接流入 |
| `branded_search_clicks` | ブランド検索からの流入 |
| `organic_search_clicks` | オーガニック検索からの流入 |
| `email_clicks` | メールからの流入 |
| `referral_clicks` | 参照元からの流入 |
| `all_other_clicks` | その他の流入 |

## データの特徴

1. **マルチブランド・マルチ地域**: 異なる`organisation_id`と`territory_name`の組み合わせで複数の時系列
2. **スパースなチャネル**: 未使用チャネルは空欄（すべてのブランドが全チャネルを使用するわけではない）
3. **時系列の長さ**: 各時系列は最低449日の連続データ
4. **通貨**: `territory_name = "All Territories"` の場合は主要地域の通貨を使用

## 用途

このデータセットはHill mixture MMMモデルの検証に適しています：
- 複数の広告チャネルの効果測定
- 飽和効果（saturation）のモデリング
- 残存効果（adstock/carryover）の推定
- 新規顧客獲得 vs 全体売上の分析

## 出典

Conjura eCommerce Analytics Platform
