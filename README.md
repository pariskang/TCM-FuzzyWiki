# TCM-FuzzyWiki V5.0

基于 XLSX 古籍章节的中医模糊知识 Wiki 构建框架。项目实现了 **observation-first** 的 MVP 链路：每一行章节先变成可追溯的 `SourceUnit`，LLM 或离线规则抽取器只抽取 observation，不直接输出证候结论；后续由本体映射、模糊集合交叠积分、共现模式挖掘、专家规则、Larsen-style 推理、相关性折扣分层聚合和 Markdown Wiki 生成器完成可审计知识编译。

> 形式化边界：本项目生成的 μ 值是基于文本证据、语言变量映射、模糊规则和专家校准配置得到的形式化近似结果，不等同于现代临床诊断。

## 已实现能力

- **XLSX/CSV 导入**：支持一行一个章节，并保留书名、章节、朝代、作者、主题、流派、地域、学术传统、来源权威性、文本完整性和语义清晰度等 metadata。
- **章节级证据单元**：把原始章节转换为 `SourceUnit`，仅保存证据和质量权重，不在该层做诊断判断。
- **llmlite 架构**：`tcm_fuzzywiki.llmlite.ChatModel` 是极简 LLM 协议；内置 Azure ChatGPT REST 适配器，也提供离线 deterministic extractor 便于测试与冷启动。
- **Observation-first 抽取**：LLM 只允许返回 `feature / feature_value / evidence_text / extraction_confidence`，禁止直接输出证候、病机或诊断。
- **Observation 标准化与未映射日志**：通过 `observation_mapping` 映射标准 observation，并把高置信未覆盖项写入 `unmapped_observations.log`。
- **Bootstrap prior 词表、专家校准与 LLM 分角色打分**：配置中每个语言变量带 `status`、`icc` 和 `review_status`；`tcm-fuzzywiki calibrate` 可汇总专家 membership CSV，`tcm-fuzzywiki roleplay-score` 可让现代循证医学专家/中医专家/古文字专家三个 LLM 角色自动打分并生成 calibrated YAML。
- **交叠积分隶属度**：当某 linguistic value 配置了 `linguistic_set` 且目标 fuzzy variable 存在对应 fuzzy set 时，使用 `∫ μ_linguistic(x)·μ_target(x) dx / ∫ μ_linguistic(x) dx`（trapezoidal 数值积分），`memberships.csv` 的 `calculation_mode` 记为 `overlap_integral`；若未配置 `linguistic_set`，则回退到配置中的 bootstrap `prior_membership` 常量先验，并把 `calculation_mode` 如实记为 `prior_membership`，避免把常量先验误标为积分结果。
- **低 ICC 不确定性传播**：当 `icc < 0.75` 时进行 Monte Carlo 采样，输出 `p5 / p95 / uncertainty_width`；点估计 `membership` 仍保持确定性计算值（积分或先验），不会被采样均值覆盖，且按映射独立 seed，保证相同映射在不同运行/顺序下可复现。
- **Observation 共现模式挖掘**：生成章节 itemset、support、confidence、lift、PMI、Jaccard、source diversity、tradition entropy、来源列表与代表证据。
- **规则生命周期入口**：输出 `candidate_patterns.csv` 与 `expert_rule_review.csv`，用于专家把候选共现模式升格为正式规则。
- **Larsen-style weighted activation**：规则激活公式为 `α = rule_weight × ∏ μ_i ^ w_i`，且前件统一读取 fuzzy variable membership。
- **三层聚合**：章节内规则质量折扣加权均值、传统内来源质量加权均值、跨传统可配置 δ 折扣 noisy-or，并在聚合表中记录折扣参数。
- **Markdown Wiki**：自动生成 `index.md`、章节页、observation 页、entity/syndrome 页、candidate pattern 页、规则页、传统页、综合谱系页和 audit 页面。
- **实现审计、完整性评估与复现 Manifest**：生成 `implementation_audit.csv`、`completion_assessment.csv`、`run_manifest.json` 及对应 Wiki 页，明确回答该次构建是 `formal_ready`、`research_ready_with_caveats` 还是 `not_ready`，并记录输入/config SHA256。
- **关系网络导出**：生成 `relation_nodes.csv` 与 `relation_edges.csv`，表达 source→observation→fuzzy variable→rule→conclusion→global conclusion 的可审计网络。
- **Mamdani 敏感性分析**：可选计算 consequent fuzzy set 截断、max 聚合与 centroid 去模糊结果，输出 `mamdani_results.csv` 和 Wiki 审计页。
- **评估指标框架**：实现 FCR、CRP、MIC、SMB、FIA-local、FIA-chain 的公式与金标准 CSV 模板；无专家 gold 时输出 `needs_gold_standard`，有 gold 时直接计算。
- **完备性验证**：生成 `validation_report.csv` 和 `wiki/audit/validation_report.md`，并提供 `tcm-fuzzywiki doctor` 命令检查配置与输入 metadata 是否达到正式分析要求。

## 安装

```bash
python -m pip install -e '.[dev]'
```

## 快速运行 Demo

```bash
tcm-fuzzywiki run-demo --output build/demo
```

Demo 输入位于 `examples/bootstrap_chapters.csv`，输出包括：

```text
build/demo/
  data/
    source_units.csv
    source_metadata.csv
    observations.csv
    memberships.csv
    observation_itemsets.csv
    cooccurrence_stats.csv
    candidate_patterns.csv
    expert_rule_review.csv
    rules.csv
    inference_results.csv
    aggregation_results.csv
    wiki_pages.csv
    relation_nodes.csv
    relation_edges.csv
    mamdani_results.csv
    evaluation_results.csv
    evaluation_gold_templates.csv
    validation_report.csv
    implementation_audit.csv
    completion_assessment.csv
    run_manifest.json
    unmapped_observations.log
  wiki/
    index.md
    sources/
    observations/
    entities/
    syndromes/
    traditions/
    rules/
    patterns/
    synthesis/
    audit/
```

## 配置/数据体检

```bash
tcm-fuzzywiki doctor \
  --config configs/tcm_fuzzywiki.yaml \
  --input examples/bootstrap_chapters.csv
```

`doctor` 不运行完整推理，只检查配置结构、规则 fuzzy-set 引用、输入章节 metadata、重复 source_id 和缺失文本。完整构建还会输出 `data/validation_report.csv` 与 `wiki/audit/validation_report.md`。

已有构建产物时，可以用 `assess` 明确判断是否“完美/正式就绪”：

```bash
tcm-fuzzywiki assess --output build/demo
```

输出 verdict：`formal_ready` 表示该次构建未发现阻塞、警告、缺失 gold 或 MVP 边界；`research_ready_with_caveats` 表示代码链路可运行但仍有研究边界；`not_ready` 表示存在阻塞问题。

## 从自己的 XLSX/CSV 构建

```bash
tcm-fuzzywiki build \
  --input path/to/chapters.xlsx \
  --config configs/tcm_fuzzywiki.yaml \
  --output build/my_wiki

# 如已有专家金标准，可额外传入：
tcm-fuzzywiki build \
  --input path/to/chapters.xlsx \
  --config configs/tcm_fuzzywiki.yaml \
  --gold-dir path/to/gold_csv_dir \
  --output build/evaluated_wiki
```

最低建议字段：

| 字段 | 含义 |
|---|---|
| `source_id` | 唯一章节 ID；缺失时自动生成。 |
| `book_name` | 书名。 |
| `volume_name` | 卷名。 |
| `chapter_title` | 章节标题。 |
| `chapter_order` | 章节顺序。 |
| `dynasty` | 朝代。 |
| `author` | 作者。 |
| `text_original` | 古籍原文。 |
| `text_type` | 条文/医案/方论/药论/针灸论等。 |
| `topic_hint` | 初始主题。 |
| `school_tag` | 学派标签，可用分号分隔多个值。 |
| `region_tag` | 地域标签，可用分号分隔多个值。 |
| `tradition_id` | 学术传统 ID。 |
| `text_family` | 文献类型家族。 |
| `citation_family` | 引用谱系。 |
| `source_authority` | 来源权威性，0–1。 |
| `text_integrity` | 文本完整性，0–1。 |
| `semantic_clarity` | 语义清晰度，0–1。 |

不确定的 metadata 请填 `uncertain`，不要留空，便于后续专家和文献考证修订。

## Azure ChatGPT / llmlite 示例

本项目不把 LLM SDK 绑定到核心链路，而是使用轻量 `ChatModel` 协议。Azure ChatGPT 调用示例：

```bash
export AZURE_OPENAI_ENDPOINT='https://<resource>.openai.azure.com'
export AZURE_OPENAI_DEPLOYMENT='<deployment-name>'
export AZURE_OPENAI_API_KEY='<api-key>'
export AZURE_OPENAI_API_VERSION='2024-02-15-preview'

tcm-fuzzywiki build \
  --input path/to/chapters.xlsx \
  --config configs/tcm_fuzzywiki.yaml \
  --output build/azure_wiki \
  --azure-llm
```

Azure 模式下，LLM 仍然只能抽取 observation；证候隶属度、规则激活、聚合与 Wiki 结果由 deterministic pipeline 复算。

## 配置文件结构

核心配置在 `configs/tcm_fuzzywiki.yaml`：

- `membership_calculation`：默认 `overlap_integral`，配置 trapezoidal 积分点数。
- `mapping_policy`：未映射 observation 日志与覆盖率阈值。
- `uncertainty_propagation`：低 ICC Monte Carlo 参数。
- `aggregation`：source 内规则折扣 gamma、跨传统默认 δ 与传统独立性权重。
- `candidate_pattern_filter`：support、lift、PMI、source_count、tradition_count 阈值。
- `fuzzy_sets`：目标 fuzzy variables 与 fuzzy sets。
- `observation_mapping`：古籍原文表达到标准 observation 的映射。
- `linguistic_values`：bootstrap prior 语言变量词表。
- `seed_rules`：MVP 冷启动规则；正式研究中可用专家审核后的 `rules.csv` 扩展或替换。
- `entities`：中医本体词表冷启动实体。
- `mamdani_sensitivity` / `consequent_fuzzy_sets`：可选 Mamdani 敏感性分析和 consequent fuzzy set 配置。

## 实现边界审计

本仓库现在会在每次构建时输出两类审计产物：

- `data/implementation_audit.csv`：机器可读的能力状态表。
- `wiki/audit/implementation_audit.md`：面向研究者/专家的 Markdown 审计说明。
- `data/completion_assessment.csv`：对该次 build 是否 formal-ready 的机器可读 verdict。
- `wiki/audit/completion_assessment.md`：对“是否完美实现/是否正式就绪”的 Markdown 解释。
- `data/run_manifest.json`：记录输入、配置、可选规则/gold、执行模式、版本和 SHA256。
- `wiki/audit/run_manifest.md`：面向审计和复现实验的运行 manifest。
- `data/validation_report.csv`：配置、输入、覆盖率、规则和 gold 标准的质量检查。
- `wiki/audit/validation_report.md`：面向研究者的完备性与正式分析就绪度报告。

状态含义：

| 状态 | 含义 |
|---|---|
| `implemented` | 已在当前 pipeline 中完整实现并可复算。 |
| `implemented_mvp` | 已提供 MVP 或 bootstrap 工作流，但真实研究仍需要专家数据、外部本体或更多样本。 |
| `future_work` | 当前暂无对应计算模块；本版本已尽量把 V5.0 计算项转成可运行模块。 |

特别说明：Mamdani 敏感性推理与 FCR/CRP/MIC/SMB/FIA 公式已经实现；但 FCR、CRP、MIC、SMB、FIA-local、FIA-chain 的正式数值需要专家金标准 CSV。未提供 gold 时，系统输出 `needs_gold_standard` 与模板，不会凭空生成专家评估结论。

## 关系网络导出

除 Markdown Wiki 外，系统会生成：

- `data/relation_nodes.csv`
- `data/relation_edges.csv`

它们使用轻量 CSV 表达 source、observation、fuzzy variable、rule、conclusion 和 global conclusion 的关系，可导入 NetworkX、Gephi、Neo4j 或其他图分析工具。

## LLM 分角色打分与专家校准 Bootstrap Prior

如果暂时没有人工专家评分，可以先用大模型自动分角色扮演三类专家生成校准分数：

- 现代循证医学专家：从现代医学、症状学和可验证证据角度打分。
- 中医专家：从中医诊断学、证候学、寒热虚实和病机角度打分。
- 古文字/训诂专家：从古汉语语义、异名、文献语境和古籍表达习惯角度打分。

Azure ChatGPT 调用示例：

```bash
tcm-fuzzywiki roleplay-score \
  --config configs/tcm_fuzzywiki.yaml \
  --output-scores build/llm_roleplay_scores.csv \
  --output-config configs/linguistic_values.roleplay_calibrated.yaml \
  --report build/llm_roleplay_calibration_report.csv \
  --azure-llm
```

离线演示/测试可使用 deterministic demo：

```bash
tcm-fuzzywiki roleplay-score \
  --config configs/tcm_fuzzywiki.yaml \
  --output-scores build/roleplay_demo_scores.csv \
  --output-config build/roleplay_demo_calibrated.yaml \
  --report build/roleplay_demo_report.csv \
  --offline-demo
```

输出 CSV 与人工专家评分格式兼容，并额外保留 `expert_role`、`confidence`、`rationale`、`score_source` 以便审计。LLM 分角色打分可用于替代冷启动阶段人工打分，但系统会保留 `calibration_source: llm_roleplay_panel`，便于后续人工复核。

人工专家也可以按如下格式准备 CSV：

| term | variable | fuzzy_set | expert_id | score |
|---|---|---|---|---:|
| 冷痛 | cold_property | high | EXP_001 | 0.90 |
| 冷痛 | cold_property | high | EXP_002 | 0.86 |

运行：

```bash
tcm-fuzzywiki calibrate \
  --config configs/tcm_fuzzywiki.yaml \
  --expert-scores path/to/expert_membership_scores.csv \
  --output-config configs/linguistic_values.calibrated.yaml \
  --report build/calibration_report.csv
```

校准会写入 `calibrated_membership`、专家均值、p5/p95、ICC/一致性 proxy、`expert_count` 和 `review_status: expert_reviewed`。低一致性条目标记为 `expert_calibrated_low_icc`，后续 membership 阶段会继续触发低 ICC 不确定性传播。

## Mamdani 与评估指标

Mamdani 敏感性分析默认启用，输出：

- `data/mamdani_results.csv`
- `wiki/audit/mamdani_sensitivity.md`

评估框架输出：

- `data/evaluation_results.csv`
- `data/evaluation_gold_templates.csv`
- `wiki/audit/evaluation_metrics.md`

可选 gold CSV 文件包括：

| 文件 | 关键字段 |
|---|---|
| `expert_memberships.csv` | `source_id,standard_observation,variable,fuzzy_set,expert_membership` |
| `expert_inference.csv` | `source_id,consequent_entity,expert_membership` |
| `expected_interpretations.csv` | `source_id,expected_consequents` |
| `modern_mappings.csv` | `entity_name,expected_modern_mapping` |
| `conditional_relations.csv` | `condition,model_membership,expert_membership` |
| `chain_paths.csv` | `case_id,model_path,expert_path,path_overlap` |

## 专家审核工作流

1. 运行 `tcm-fuzzywiki build` 生成 `candidate_patterns.csv` 和 `expert_rule_review.csv`。
2. 专家查看共现 observation、代表来源、代表证据、support、lift、PMI、流派分布与系统建议解释。
3. 专家补充 consequent、rule_weight、适用场景、冲突说明与审核状态。
4. 将专家审核结果整理为 `rules.csv` 后再次运行：

```bash
tcm-fuzzywiki build \
  --input path/to/chapters.xlsx \
  --config configs/tcm_fuzzywiki.yaml \
  --rules-csv path/to/rules.csv \
  --output build/reviewed_wiki
```

## 开发与测试

```bash
python -m pytest -q
```

当前测试覆盖：

- 交叠积分对重叠/不重叠 fuzzy sets 的基本行为。
- Demo pipeline 是否生成 observations、memberships、candidate patterns、rules、inference results、Mamdani sensitivity、evaluation outputs、relation network、validation report、implementation audit 和 Markdown Wiki。
- 专家审核 CSV 中 pending 行跳过、accepted 行升格为规则的鲁棒加载行为。
- Mamdani centroid 输出与 gold CSV 驱动的 FCR 计算。
- 配置/输入 validation 与 `doctor` 命令输出。
- LLM/离线分角色专家打分与 roleplay calibration 输出。
- `completion_assessment` 对缺失 gold、MVP 边界和 validation warnings 的 verdict。
- `run_manifest` 记录输入/config 哈希、执行模式和 summary，保证可复现实验审计。

## 目录说明

```text
tcm_fuzzywiki/
  aggregation.py     # 章节/传统/全局三层聚合
  cli.py             # 命令行入口
  cooccurrence.py    # observation itemset 与 candidate pattern 挖掘
  config.py          # YAML 配置读取
  assessment.py      # build 输出完整性 verdict
  calibration.py     # 专家 membership 校准 bootstrap prior
  extraction.py      # LLM/离线 observation 抽取与标准化
  inference.py       # Larsen-style weighted activation 推理
  io.py              # XLSX/CSV 输入与数据表输出
  llmlite.py         # 轻量 LLM 协议与 Azure ChatGPT 适配器
  membership.py      # 交叠积分与低 ICC Monte Carlo
  mamdani.py         # Mamdani 敏感性分析
  evaluation.py      # FCR/CRP/MIC/SMB/FIA 指标框架
  models.py          # 核心数据模型
  pipeline.py        # 端到端 orchestration
  rules.py           # seed/expert rules 加载
  roleplay.py        # LLM 分角色专家自动打分
  network.py         # Fuzzy relation network CSV 导出
  validation.py      # 配置/输入/运行质量验证
  audit.py           # V5.0 能力实现审计
  wiki.py            # Markdown Wiki 生成
configs/
  tcm_fuzzywiki.yaml # V5.0 默认配置
examples/
  bootstrap_chapters.csv
```
