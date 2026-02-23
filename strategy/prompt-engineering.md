# WorldFlux リポジトリ分析のためのプロンプトエンジニアリング実践ガイド

## Context

WorldFlux (v0.1.0 Alpha) のエンジニアリング・ビジネス両面の調査を、Agent Teams + Web検索を活用して効率的に行うためのプロンプトエンジニアリング手法の体系的まとめ。ソロファウンダーのPre-Seed準備段階における意思決定を加速するための実用リファレンス。

---

## I. エンジニアリング調査のプロンプト手法

### 1. Context-First Prompting（コンテキスト先行型）

既存のアーキテクチャパターンを先に提示し、それに対して推論させる手法。「このコードを分析して」ではなく、既存の規約を提示してそれに照らして分析させる。

```
あなたは以下の構造を持つPython MLフレームワークを分析しています:
- Core API: src/worldflux/ (factoryパターン、registryベースのmodel dispatch)
- 本番モデル: DreamerV3, TD-MPC2（parity proof済み）
- エントリポイント: CLI (typer), Python API (create_world_model())

既存パターン:
- [factory.pyのregistration patternを貼付]
- [config解決チェーンを貼付]

タスク: CLI → config解決 → model instantiation → training loop の
依存グラフをマッピングせよ。各エッジについて、依存が確立される
正確なfile:lineを引用すること。
```

**効果**: 既存コードパターンをfew-shotとして提供 → 仮想的パターンではなく実際のアーキテクチャに基づく推論を制約。チームで40%少ない統合問題、60%少ないデバッグ時間を報告。

### 2. 出力契約型プロンプティング（Output Contract Prompting）

成功基準と出力スキーマを事前定義する。2026年のコンセンサスでは「最も効果的な単一手法」。

```
このPythonパッケージの依存関係グラフを分析せよ。

成功基準:
- エントリポイントからリーフ依存へのimportチェーンを全て追跡
- 循環依存を正確なファイルパスとともにフラグ
- 外部 vs 内部依存を区別

出力は以下のJSONスキーマに準拠:
{
  "nodes": [{"module": str, "type": "internal|external", "imports": int}],
  "edges": [{"from": str, "to": str, "import_type": "direct|transitive"}],
  "cycles": [{"path": [str], "severity": "error|warning"}]
}
```

### 3. 制約駆動型分析（Constraint-Driven Analysis）

分析がやってはいけないことを明示し、ハルシネーションを低減する。

```
ReplayBuffer実装のスレッドセーフティ問題を分析せよ。

制約:
- 明示的なlock/queueを見つけない限り、スレッドセーフ機構の存在を仮定するな
- ドキュメントを参照するな。実際のコード行のみ引用
- 判断できないメソッドには「INDETERMINATE」と記載
- ReplayBufferはスレッドセーフではない（単一writerスレッド限定）

各publicメソッドについて報告:
1. アクセスされる共有状態（行番号付き）
2. 同期プリミティブ（存在 or "NONE"）
3. 競合リスク: HIGH/MEDIUM/LOW/NONE
```

### 4. エビデンス接地型プロンプティング（Evidence-Grounded）

コード分析におけるハルシネーション防止の最重要手法。

```
接地ルール:
1. リポジトリに存在するファイルのみ参照（検証する）
2. 提供されたコード内の行番号のみ引用
3. 不確実な場合は [UNCERTAIN] を付記
4. 関数名から動作を推論するな。実装を読め
5. 「コードがXをしている」と「コードがXをすべき」を区別

禁止パターン:
- "おそらく..." → "file:lineのコードは...を示す" に置換
- "ベストプラクティスでは..." → "このコードベースは...を実装" に置換
- "一般的なMLフレームワークでは..." → 具体的エビデンスに置換
```

### 5. Chain-of-Thought変種の使い分け

| 手法 | 用途 | WorldFluxでの適用例 |
|---|---|---|
| **CoT（Chain-of-Thought）** | 順次的調査（デバッグ、コールチェーン追跡） | training loop → replay buffer → model forward passのデータフロー追跡 |
| **ToT（Tree-of-Thought）** | 設計決定（複数の実行可能パス） | config systemリファクタリングの3つの戦略比較 |
| **SoT（Skeleton-of-Thought）** | 並列分析（独立した分析ポイント） | セキュリティ監査の5-7ポイントを並列展開 |
| **GoT（Graph-of-Thought）** | 相互依存分析 | CLI→Factory→Training→Checkpointのエラーハンドリング伝播 |

**重要**: SoTは分析ポイント間に依存関係がある場合は不適切。その場合はCoTを使用。

### 6. ペルソナベース分析

ペルソナ単独ではなく、具体的指示+例と組み合わせて最大効果。

```yaml
推奨ペルソナ構成:
  architect:
    焦点: "API表面安定性、拡張ポイント、設定の合成可能性"
    出力: "結合スコア(0-10)、拡張摩擦ポイント、改善推奨"
  security:
    焦点: "torch.loadの安全性、pickle使用、シェルインジェクション"
    出力: "[{file, line, finding, severity, recommendation}]"
  performance:
    焦点: "ホットパス特定、メモリ割当パターン、I/O vs 計算バウンド"
    出力: "O()複雑度注釈とメモリ推定"
  qa:
    焦点: "テストカバレッジギャップ、CIプリセット妥当性"
    出力: "APIサーフェス別カバレッジマトリクス"
```

---

## II. ビジネス調査のプロンプト手法

### 1. 市場規模分析（TAM/SAM/SOM）

単純な「Xの市場規模は？」は汎用的で信頼性の低い数字を生む。ボトムアップ＋トップダウンの両方で交差検証する。

```
TAM:
- バイヤーセグメント別に列挙（研究者、MLエンジニア、ロボ企業...）
- 検証可能なデータソースでセグメントカウント推定
  （GitHubデベロッパーサーベイ、Stack Overflowサーベイ、OECD AI Report）
- セグメント別年間MLツーリング支出を推定
- 計算を表示: segment_count x avg_spend = segment_TAM

重要ルール:
- 全ての仮定に [ASSUMPTION] マーク
- 各推定に信頼度 (HIGH/MEDIUM/LOW) 付記
- トップダウンとボトムアップの両方で計算し、結果を比較
- データポイントを捏造するな
```

### 2. 競合分析（リポジトリベース）

公開リポジトリの観察可能シグナルから構造化分析を行う。

```
各競合リポジトリについて以下を評価:

A. トラクションシグナル:
   - GitHub stars/forks/contributors（絶対値 + 成長率）
   - PyPI月間DL（pypistats.orgデータ使用）
   - Issue open/close比率
   - 最終コミット日、リリース頻度

B. 技術的差別化:
   - サポートアルゴリズム/モデル
   - API設計哲学
   - ドキュメント品質

C. コミュニティヘルス:
   - コントリビュータ多様性（バス係数）
   - Issue応答時間
   - 学術引用数

D. ビジネスモデルシグナル:
   - ライセンスタイプと商用利用可能性
   - 企業バッキング vs 独立
```

### 3. デューデリジェンス（技術DD）

```
信頼度評価システム:
- HIGH (90-100%): ドキュメント/コードに明示的記載
- MEDIUM (60-89%): 利用可能ソースから合理的推論
- LOW (30-59%): 業界標準または部分的エビデンス

評価領域:
1. コード品質・エンジニアリング実践 → スコア1-5
2. アーキテクチャ・スケーラビリティ → スコア1-5
3. セキュリティ・信頼性 → スコア1-5
4. 技術的負債・メンテナンス → スコア1-5
5. 知的財産 → スコア1-5

各領域: | 評価領域 | スコア | 信頼度 | 根拠 | リスクレベル |
```

### 4. Red Team / Devil's Advocate

「なぜ失敗するか教えて」では浅い批判しか出ない。構造化された敵対的ワークフローが最も効果的。

**Pre-Mortem分析テンプレート**:
```
18ヶ月後。WorldFluxは失敗し停止した。ポストモーテムを書け。

構成:
1. 失敗のタイムライン（Month 1-3, 4-6, 7-12, 13-18）
2. 根本原因（寄与度順に3つ、各確率%付き）
3. 無視された警告サイン（どのメトリクスがアラームを出すべきだったか）
4. 反事実：何が会社を救ったか
5. 次の試みへの教訓

研究結果: 失敗が既に起きたと想像することで、
原因特定能力が30%向上する（prospective hindsight）
```

**Anti-Sycophancy Framing（追従防止）**:
```
重要指示: 検証を求めていない。真実を求めている。
励ますことを書いている自分に気づいたら、立ち止まって問え:
「これはエビデンスに支持されているか、それとも同調しているだけか？」
根拠なき楽観を罰し、エビデンスに基づく悲観を報いよ。
```

### 5. GTMスター予測のベンチマーキング

```
以下のML/RLフレームワークのGitHubスター成長軌跡を分析し、
新規参入者の現実的成長曲線をモデル化せよ:

参照プロジェクト:
- Stable Baselines3: ローンチ以降のスター推移
- CleanRL: ローンチ以降のスター推移
- Gymnasium (Farama): リブランド以降のスター推移

各プロジェクトについて特定:
1. ローンチから100スターまでの期間
2. 100から1,000スターまでの期間
3. 変曲点と原因（カンファレンストーク？バイラルツイート？）
4. スターとPyPIダウンロードの相関

CONSERVATIVE/BASE/OPTIMISTICの3シナリオをモデル化。
```

---

## III. Agent Teamsを活用したマルチエージェント調査パターン

### 1. 推奨アーキテクチャ: Supervisor Pattern (Leader-Worker)

Claude Code Teamsでの最も実証されたパターン。リーダーが問題分解→専門ワーカー生成→結果検証→統合レポート。

```
チーム構成例（WorldFlux投資適格性調査）:

Lead Agent (Opus):
  役割: 問題分解、ワーカー生成、結果統合
  ツール: TeamCreate, TaskCreate, TaskUpdate, SendMessage

Worker 1 - "code-architect":
  焦点: モジュール構造、依存グラフ、技術的負債定量化
  ファイル: src/worldflux/core/, src/worldflux/factory.py

Worker 2 - "security-auditor":
  焦点: 脆弱性、OWASP準拠、secrets検出
  ファイル: src/worldflux/**/*.py

Worker 3 - "quality-analyst":
  焦点: テストカバレッジ、複雑度メトリクス、ドキュメント
  ファイル: tests/**/*.py, pyproject.toml

Worker 4 - "market-analyst":
  焦点: 競合環境、市場規模、差別化
  ツール: WebSearch, WebFetch

Worker 5 - "devils-advocate":
  焦点: Phase 1統合後に全結論に挑戦
  ブロック: Worker 1-4完了後に開始
```

### 2. タスク分解と依存関係

```python
# Phase 1: 並列実行（全て独立）
TaskCreate("Code Architecture Analysis")  # task-1
TaskCreate("Security Audit")              # task-2
TaskCreate("Quality Analysis")            # task-3
TaskCreate("Market/Business Analysis")    # task-4

# Phase 2: Phase 1に依存
TaskCreate("Cross-Domain Synthesis")      # task-5
TaskUpdate(taskId="5", addBlockedBy=["1","2","3","4"])

# Phase 3: Phase 2に依存
TaskCreate("Adversarial Review")          # task-6
TaskUpdate(taskId="6", addBlockedBy=["5"])
```

### 3. 結果統合テクニック

各ワーカーに統一テンプレートを与えることで統合を劇的に容易にする:

```
各ワーカーの出力テンプレート:
1. サマリースコア (1-5)
2. 主要発見 (severity: CRITICAL/HIGH/MEDIUM/LOW)
3. リスク (probability + impact)
4. 推奨事項 (effort estimates付き)
5. 他ドメインとの依存関係
```

**Hierarchical Aggregation**: リーダーが全レポートを収集し交差参照。単一ワーカーが見つけない**ドメイン横断リスク**を発見する核心的価値。

### 4. 実装チェックリスト

1. `TeamCreate` でチーム作成
2. `TaskCreate` で全タスクを作成、`addBlockedBy` で依存関係設定
3. `Task(subagent_type="general-purpose", team_name=..., name=...)` でワーカー生成
4. リーダーはOpus、ワーカーはSonnet（コスト最適化）
5. ファイル/ディレクトリの所有権境界を明確に割り当て（競合回避）
6. 各ワーカーに構造化出力テンプレートを含める
7. `SendMessage` で専門家間の調整
8. 全ワーカーに `shutdown_request` 後 `TeamDelete`

---

## IV. メタプロンプティングパターン（横断的手法）

### 1. 信頼度校正出力

全ての定量的主張に信頼度を付記させる。市場規模・財務予測で特に重要。

```
全ての定量的主張に付記:
[信頼度: HIGH/MEDIUM/LOW] [ソース: cited/estimated/assumed]
```

### 2. 敵対的交差検証

同じ分析を2つの異なるフレーミングで実行し結果を比較。

```
Prompt A: "あなたはポテンシャルを見出す強気VCパートナー..."
Prompt B: "あなたは過去に痛い目を見た弱気VCパートナー..."
統合: "2つの分析を比較。一致点はどこか？不一致点では
どちらのポジションがより強いエビデンスを持つか？"
```

### 3. 類似企業アンカリング

```
推定を提供する前に、類似段階の3-5の類似プロジェクトを特定。
各類似について:
- 名前、比較時の段階
- その段階での主要メトリクス
- WorldFluxの同メトリクスとの比較
- 調整係数と根拠
```

### 4. 段階ゲート型プロンプティング

```
この分析には4段階ある。各段階が完了・検証されるまで次に進むな。

GATE 1: データ収集（全ソース列挙、信頼度マーク）
GATE 2: 分析（フレームワーク適用、計算表示）
GATE 3: 統合（結論導出、矛盾特定）
GATE 4: 推奨事項（優先度付き、実行可能、期限付き）

各ゲートで: "GATE X COMPLETE. GATE X+1に進む。"
完了できない場合: "GATE X BLOCKED: [理由]"
```

### 5. 差分分析（Ground Truth Anchoring）

最も実用的な手法。機械的に検証可能な事実とLLM分析を比較。

```
Step 1: 機械的分析ツールを実行
  ruff check src/       → 実際のlint結果
  mypy src/worldflux/   → 実際の型エラー
  pytest --co tests/    → 実際のテスト一覧

Step 2: ツール出力をground truthとしてLLMに提供
  "以下は実際のmypy結果。各エラーについて:
   1. 根本原因を説明（関連する型定義を引用）
   2. 重大度を評価（blocking vs cosmetic）
   3. 後方互換性を維持する修正を提案"

Step 3: LLMの主張をツール出力と交差検証
  ツールが発見しない問題をLLMが報告 → [NEEDS VERIFICATION]
```

---

## V. WorldFlux固有の実践プロンプト集

### プロンプト1: パリティプルーフ手法の学術的新規性評価

```
あなたはML検証手法の専門家。以下のパリティプルーフ手法を評価:
- TOST（Two One-Sided Tests）+ Bootstrap信頼区間
- Bayesian HDI（Highest Density Interval）

質問:
1. MLフレームワーク空間でこの手法を提供している競合は存在するか？
2. 学術論文としての新規性はどの程度か？
3. EU AI Act (2026年8月施行) との関連性は？
4. NeurIPS/ICMLワークショップ投稿の実現可能性は？

各回答に[信頼度]と[ソース]を付記。
```

### プロンプト2: Factory APIの競合優位性評価

```
WorldFluxの `create_world_model("dreamerv3:size12m")` APIを
以下の類似プロジェクトのAPIと比較:
- HuggingFace: `AutoModel.from_pretrained("bert-base")`
- Stable Baselines3: `PPO("MlpPolicy", "CartPole-v1")`
- CleanRL: スクリプトベース（APIなし）

評価軸:
1. Time-to-First-Value（最初の有意義な結果まで）
2. 認知的負荷（新規ユーザーの学習曲線）
3. 拡張性（新モデル追加の摩擦）
4. エコシステム統合（既存ツールとの互換性）
```

### プロンプト3: ソロファウンダーリスク評価

```
Pre-Seed段階のソロファウンダーMLスタートアップのリスクを評価。
定量的エビデンスを使用:

1. ソロファウンダー成功率 vs 共同創業チーム（YCデータ引用）
2. MLインフラ企業のファウンダー典型プロファイル
3. ~58K行のコードベースが示す実行力の信号強度
4. バス係数=1の具体的リスクシナリオ
5. 緩和策: アドバイザーボード、初期採用者コミュニティ

各ポイントに [ASSUMPTION] or [DATA] マーク。
```

---

## VI. 主要ソース

### エンジニアリング側
- Anthropic: Multi-Agent Research System (Jan 2026)
- ACM TOSEM: Structured Chain-of-Thought for Code Generation
- arxiv 2512.12117: Citation-Grounded Code Comprehension
- OpenSSF: Security-Focused Guide for AI Code Assistant Instructions
- Lakera: The Ultimate Guide to Prompt Engineering in 2026

### ビジネス側
- Rebel Fund: AI-Driven Due Diligence Checklist
- McKinsey: Using Gen AI for Outside-In Diligence
- Decibel VC: Developer Marketing Early-Stage Playbook
- MarketsandMarkets: AI Platform Market Forecast to 2030
- Qubit Capital: AI Startup Fundraising Trends 2026

### Agent Teams
- Anthropic: Claude Code Agent Teams Documentation
- Deloitte: Unlocking Value with AI Agent Orchestration
- DataCamp: CrewAI vs LangGraph vs AutoGen Comparison
- Microsoft Azure: AI Agent Design Patterns
