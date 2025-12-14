# Phase3実装: 完全ノンブロッキング・フェーズベースアーキテクチャ

## 問題の背景

### Phase2の課題
Streamlit Cloud環境で `ThreadPoolExecutor` + `future.result(timeout)` を使用した split単位 watchdog が、
**1/3 split（約45〜50秒）で UI が固まる問題**が発生。

**原因:**
- Gemini API 成功後〜JSON guard 開始で制御が戻らない
- timeout / exception ログも出ない
- Thread 内での I/O or C拡張ブロックが原因と判断

## Phase3 解決策: フェーズベースアーキテクチャ

### 設計原則

**絶対ルール:**
1. **ThreadPoolExecutor / multiprocessing / future.result(timeout) を完全に廃止**
2. **1 rerun = 1 フェーズを厳守** - forループで複数フェーズを回さない
3. **全フェーズはメインスレッドで実行** - 必ず制御が戻る
4. **各フェーズ完了後は必ず st.session_state に保存 → st.rerun()**

### アーキテクチャ変更

#### 旧構造（Phase2）:
```
PROCESSING phase
  └─ process_single_split (ThreadPoolExecutor)
      └─ future.result(timeout=120)  ← ここでブロック
          └─ extractor.extract_with_retry
              ├─ Gemini API call (~40s)
              ├─ JSON parsing
              └─ Post-processing
```

#### 新構造（Phase3）:
```
PROCESSING phase
  ├─ split 0
  │   ├─ GEMINI_CALL    → st.rerun()
  │   ├─ JSON_PARSE     → st.rerun()
  │   ├─ POSTPROCESS    → st.rerun()
  │   └─ VALIDATION     → st.rerun()
  ├─ split 1
  │   ├─ GEMINI_CALL    → st.rerun()
  │   └─ ...
  └─ MERGE              → st.rerun()
```

**各 rerun で1フェーズのみ実行 → 必ず制御がUIに戻る**

## 実装詳細

### 新規ファイル

#### 1. `utils/split_phases.py`
Split-level の詳細フェーズ定義:

```python
class SplitPhase(Enum):
    GEMINI_CALL = "gemini_call"    # Gemini API呼び出し
    JSON_PARSE = "json_parse"      # JSON パース
    POSTPROCESS = "postprocess"    # 前段整形
    VALIDATION = "validation"      # 貸借ペア保証・金額検証
    COMPLETED = "completed"        # 完了
    FAILED = "failed"              # 失敗
```

`SplitProcessingState`: 各分割の処理状態を保持
- `gemini_response`: Gemini APIレスポンス
- `parsed_json`: パース済みJSON
- `processed_data`: 前段整形後データ
- `validated_data`: 検証済み最終データ

#### 2. `backend_processor_phase.py`
完全フェーズベースのプロセッサ:

```python
class PhaseBasedProcessor:
    def process_phase(self, split_state, split_path, total_splits) -> Dict:
        # 現在のフェーズに応じて処理を分岐
        if split_state.phase == SplitPhase.GEMINI_CALL:
            return self._phase_gemini_call(...)
        elif split_state.phase == SplitPhase.JSON_PARSE:
            return self._phase_json_parse(...)
        # ...以下略
```

**各フェーズメソッド:**
- `_phase_gemini_call()`: Gemini API呼び出し（メインスレッド）
- `_phase_json_parse()`: JSON パース（utils.json_guard使用）
- `_phase_postprocess()`: 前段整形（utils.reconcile_entries使用）
- `_phase_validation()`: 貸借ペア保証・金額検証（utils.postprocess使用）

### 更新ファイル

#### 3. `utils/processing_phases.py`
```python
@dataclass
class ProcessingState:
    # Phase3追加フィールド
    split_states_data: List[Dict[str, Any]]  # Split phase states
    phase_stall_count: int = 0  # フェーズ停滞カウンタ
    max_phase_stall: int = 5    # 停滞検出閾値
```

#### 4. `app.py`
PROCESSING フェーズを完全書き換え:

```python
# 現在のsplit処理状態を取得または作成
if st.session_state.current_split_state is None:
    st.session_state.current_split_state = SplitProcessingState(...)

split_state = st.session_state.current_split_state

# 1フェーズだけ処理
result = processor.process_phase(
    split_state=split_state,
    split_path=split_path,
    total_splits=state.total_splits
)

# フェーズ完了 → st.rerun()
if result["split_complete"]:
    # 次のsplitへ
    state.current_split_index += 1
    st.session_state.current_split_state = None
    st.rerun()
else:
    # 次のフェーズへ
    st.rerun()
```

## タイムアウト管理の変更

### 旧方式（Phase2）:
```python
# 時間ベース
timeout_seconds = 120
future.result(timeout=timeout_seconds)  # ブロックする
```

### 新方式（Phase3）:
```python
# 進捗ベース
phase_stall_count: int = 0
if state.is_phase_stalled():  # 5回連続停滞で中断
    state.phase = ProcessingPhase.ERROR
```

**理由:**
- メインスレッド実行なので時間ベースタイムアウトは不要
- 進捗がない場合（同じフェーズで停滞）を検出して中断

## 期待される効果

### ✅ UI固まり完全解消
- **全フェーズがメインスレッド実行** → 必ず制御がUIに戻る
- **1 rerun = 1 フェーズ** → 各フェーズ後にUI更新

### ✅ 100+ページ確実完走
- **各フェーズは独立** → 1フェーズが重くても次のrerunで必ず戻る
- **進捗ベースタイムアウト** → 停滞を確実に検出

### ✅ 詳細な進捗表示
```
📄 分割 1/10: 🤖 Gemini API 呼び出し中
📄 分割 1/10: 📊 JSON パース中
📄 分割 1/10: 🔧 データ後処理中
📄 分割 1/10: ✅ データ検証中
```

### ✅ エラー可視性向上
- 各フェーズでのエラーを明確に表示
- 「このフェーズで停止しました」を必ず表示

## 禁止事項

以下を**絶対に使用しない**:
- `ThreadPoolExecutor`
- `multiprocessing`
- `future.result(timeout)`
- `concurrent.futures.TimeoutError`
- OS signal / kill

## テスト計画

### 1. ローカル環境テスト
```bash
streamlit run app.py
```
- 小規模PDF（5ページ）でフェーズ遷移を確認
- 中規模PDF（30ページ）で全フェーズ完走を確認

### 2. Streamlit Cloud デプロイテスト
- 大規模PDF（100+ページ）で完走確認
- 47秒・1/3で固まる現象が完全に消えるか確認
- 各フェーズでUI更新されるか確認

### 3. エラーハンドリングテスト
- Gemini APIエラー発生時の挙動
- JSON parseエラー発生時の挙動
- フェーズ停滞検出の挙動

## マイグレーションパス

既存の `backend_processor_stepwise.py` は残したまま、
新しい `backend_processor_phase.py` を使用:

```python
# app.py
if st.session_state.phase_processor is None:
    st.session_state.phase_processor = PhaseBasedProcessor()  # 新
```

ロールバック時は:
```python
# app.py
if st.session_state.stepwise_processor is None:
    st.session_state.stepwise_processor = StepwiseProcessor()  # 旧
```

## まとめ

**Phase3は「絶対にUIが止まらない」構造**:
1. ThreadPoolExecutor完全廃止 → メインスレッド実行
2. 1 rerun = 1フェーズ → 必ず制御が戻る
3. 進捗ベースタイムアウト → 停滞を確実に検出

これにより**47秒・1/3で固まる現象が完全に消える**はずです。
