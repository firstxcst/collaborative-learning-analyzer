# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [1.1.0] - 2026-10-09

一次针对审计问题的全面修复。**包含破坏性变更**：报告字段与计分公式均不向后兼容。

### 破坏性变更

- **计分公式重写**。`均衡度 = 1 - 2×std(贡献度)` 改为 HHI 归一化反向值，
  并引入乘性活跃度门控：`health = 100 × activity × quality`。
  旧公式下全员沉默的小组得 90.5 分（excellent），且 `critical` 等级在数学上不可达。
- **报告字段重命名/新增**：
  `balance_score` → `evenness`；`collaboration_mode_score` 拆分为
  `quality_score` + `interaction_depth`；新增 `participation_activity`、
  `turn_taking_pattern`、`consensus_quality`、`data_completeness`、`warnings`。
- **可选指标改为 `Optional[float]`**。`topic_relevance` / `interaction_depth` /
  `consensus_quality` / `cohesion_score` 在不可用时为 `null`，不再用 `0.5` 冒充。
- **安装后的顶层包名变更**：由 `src` 改为 `collaborative_learning_analyzer`。
- **核心依赖清空**：`pip install` 不再拉取 torch；按模态拆分为 extras。
- `group_id` 默认值由 `group_1` 改为调用方显式传入（CLI 默认 `group_1`）。

### 修复

**计分**
- 全员沉默不再得到高分：现在恒为 0 分（critical）
- `critical` 等级重新可达，0–100 全区间可用
- 个体语义分改为**逐人**取得，不再把全局分数复制给每个成员
- 缺失模态改为在可用维度上重新归一化权重，不再填充默认值
- 权重集中配置并校验，写错会立即报错

**视觉**
- `_compute_attention_map` / `_detect_pointing` / `_analyze_gaze` 三个**空方法**给出真实实现
- 补齐被调用但从未定义的标注视频输出
- 帧率改为从容器真实读取（旧版硬编码 30fps，25fps 素材会导致时间轴偏差 20%）
- 改为流式处理，内存占用不再随视频长度线性增长
- `cohesion_score` 不再硬编码兜底 0.5

**语义**
- 修复 LLM 返回 `null` / 非数字时抛未捕获异常导致整条流水线崩溃
- **修复解析失败静默降级为 0.5 分**：现在计入失败窗口并写入告警
- 冗长转录由「取头尾各 2000 字符」改为**滑窗分块**，不再丢弃中段
- 修复 dashscope 分支引用了不存在的配置字段（必然 AttributeError）
- 补齐超时、重试、最大输出 token、并发上限
- `turn_taking_pattern` 与 `consensus_quality` 不再"算了不用"

**语音**
- 修复 `whisper_device="auto"` 导致默认配置开箱即崩（`torch.device("auto")` 会抛错）
- 文本归属改为最大重叠分配，不再把同一句话重复分配给相邻说话人
- 声纹注册与身份对齐从死代码变为可用功能（`--align-profiles`）
- 新增转录导入（分会场麦克风 / 教师字幕 / 外部 ASR 结果）

**工程与打包**
- 移除 4 个模块顶层的 `warnings.filterwarnings("ignore")`（会污染宿主进程）
- 移除导入期副作用（不再在 import 时创建目录）
- 修复非 editable 安装后 `PROJECT_ROOT` 指向 site-packages 导致 `.env` 读不到
- 注册 `cla` 命令行入口；补齐 `python -m` 入口
- 修复 `GroupCollaborationReport.load()` 丢失个体贡献度与模态明细
- 修复顶层发布名为 `src` 的命名空间污染
- 清理 13 处未使用导入与 3 个从未被引用的定义，并加自动化回归测试

**文档与合规**
- 许可证统一为 AGPL-3.0-or-later，并换用 GNU 官方全文
  （此前 README / LICENSE / CONTRIBUTING 对许可证有四种互相矛盾的说法）
- 隐私说明改为如实区分「音视频不出本地」与「转录会发送给所配置的 LLM」
- 移除对"加密存储、定期删除"等**没有实现**的能力承诺
- 移除把自评包装成"社区评审报告"的文档，以及含过期声明的项目总结
- 修复快速开始引用不存在的 `data/sample.wav`（该文件被 gitignore，不可能存在）；
  改为由 `tools/make_fixtures.py` 生成真实媒体夹具

### 新增

- `tools/make_fixtures.py`：生成**真实媒体**夹具（SAPI 合成语音 + 真实时间轴 + 真实视频），
  并显式声明其局限
- `tests/`：完整测试套件（计分 / 数据模型 / 语义 / 语音 / 视觉 / 打包 / 端到端）
- `docs/SCORING_MODEL.md`：计分模型推导与「权重尚未实证标定」的说明
- `docs/MUTATION_TESTING.md`：变异测试结果
- `docs/AUDIT_REMEDIATION.md`：审计问题逐条修复对照（含 2 条撤回）
- `cla` CLI：`analyze` / `paths` 子命令
- GitHub Actions CI

### 已知问题

- 权重与阈值是工程默认值，**未经真实课堂数据标定**；效度证据缺失
- 视线估计为几何启发式，侧面机位与遮挡下误差大，**无精度评估**
- 不提供数据生命周期管理

## [1.0.0] - 2026-05-01

初始公开版本。包含语音/视觉/语义三个智能体与融合引擎的骨架，
以及 README、示例脚本与文档。**该版本的计分逻辑存在严重缺陷**（见 1.1.0 说明），
且视觉模块的三个核心分析方法是空实现。
