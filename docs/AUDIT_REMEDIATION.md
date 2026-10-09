# 审计问题修复对照表

本文件逐条记录审计指出的问题、修复方式与验证证据。
其中 **2 条结论经复核后撤回**，1 条修正 —— 一并列出，不做隐藏。

## 0. 首先：撤回与修正

| 编号 | 我当时的结论 | 复核结果 |
|------|--------------|----------|
| #55 | `REVIEW_REPORT.md` 引用的 commit `cd931e1` / `996e412` 不存在，是伪造的评审记录 | **撤回**。两个 hash 都真实存在，分别对应「完善代码和文档」与「添加社区贡献所需文件」两次提交。当时的判断基于 `--depth 1` 浅克隆，属方法错误 |
| #68 | 24 个文件来自同一次提交 | **撤回**。仓库有 8 次提交，首次提交 16 个文件 |
| #46 | 单一提交 + 个人 git 身份 | **修正**为：8 次提交全部发生在 2026-05-01 的 15:21–15:57，**36 分钟之内** |
| #55（保留部分） | —— | 仍然成立的部分：`REVIEW_REPORT.md` 把自评包装成「社区评审报告」并给 4/5 自评分，同时自己承认核心模块无测试。该文件已移除 |

## 1. 计分引擎（#1–#9）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 1 | `均衡度 = 1-2σ`，全员沉默得 90.5 分（excellent） | 改为 HHI 归一化反向值；沉默返回 0 | `test_evenness_of_silent_group_is_zero_not_one`、`test_silent_group_gets_critical_and_zero` |
| 2 | 低分段度量反向相关 | 均衡度只对**份额分布**敏感 | `test_evenness_is_scale_invariant` |
| 3 | 可达下界 ≈32.2，`critical` 是死档 | 加乘性活跃度门控，全区间可达 | `test_all_five_levels_are_reachable`（逐级断言）、`test_score_range_is_fully_reachable` |
| 4 | 贡献度上限仅 0.3 → σ 被压缩 | 个体维度重构为 4 个真实维度 | `test_individual_scores_differ_when_participation_differs` |
| 5 | 个体语义分 = 全局分数 | 新增 `SemanticAnalysisResult.per_speaker`，逐人回填 | `test_individual_semantic_scores_are_per_person` |
| 6 | 权重无依据 | 集中在 `FusionEngineConfig` 并 `validate()` 校验；并**明确声明尚未实证标定** | `test_config_validation_rejects_bad_weights`、`docs/SCORING_MODEL.md` 第三节 |
| 7 | 单位混用（0–1 说成百分比） | 报告字段语义统一，`participation_activity` 等明确为 0–1 | `test_end_to_end_over_real_fixture` |
| 8 | 「标准差范围 0–0.5」是未验证假设 | 该公式已被整体移除 | 代码中不再出现 |
| 9 | `1-2σ` 中系数 2 无来源 | 同 #8 | —— |

## 2. 视觉模块（#10–#20）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 10 | `_compute_attention_map` / `_detect_pointing` / `_analyze_gaze` 三个方法体为空 | 合并为 `_observe_interactions`，给出真实几何判定实现 | `test_gaze_event_created_when_head_points_at_other_person`、`test_pointing_event_when_wrist_near_target` |
| 11 | 事件容器恒为空 | 事件聚合器输出真实区间 | 同上 |
| 12 | `cohesion_score` 硬编码 0.5 | 未分析时返回 `None` | `test_cohesion_is_none_without_data`、`test_video_result_defaults_are_not_fabricated` |
| 13 | `_save_annotated_video` 被调用但未定义 | 补齐为 `_draw_debug_frame` + `_open_writer` | `test_annotated_video_writer_creates_file` |
| 14 | `model.track()` 生成器耗尽后复用 | 改为逐帧流式处理 | `test_cv2_is_not_imported_at_module_level` 等 |
| 15 | 时间戳硬编码 `i/30` | 从容器读真实 FPS，读不到才兜底并标记 `fps_is_fallback` | `test_read_video_meta_reads_real_fps` |
| 16 | 全帧关键点驻留内存 | 改为流式聚合，内存与视频长度无关 | 代码结构；`frame_stride` / `max_processed_frames` 可配 |
| 17 | `analyze_pose` 无条件调用后丢弃 | 与检测合并为单次推理 | 同上 |
| 18 | `cv2` 导入未使用 | 现在真正用于解码与标注绘制 | `test_cv2_is_not_imported_at_module_level` |
| 19 | `supervision` 仅死分支导入 | 依赖与导入一并移除 | `pyproject.toml` |
| 20 | 声纹注册是死代码 | `align_speakers()` 接入流水线（`--align-profiles`） | `test_load_transcript_*` 系列 + 流水线 |

## 3. 语义模块（#21–#29）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 21 | 只捕 `JSONDecodeError`，null/中文直接崩 | 类型安全转换，非法值 → `None`，不抛未捕获异常 | `test_parse_response_returns_none_for_null_fields`、`test_parse_response_accepts_numeric_strings` |
| 22 | **解析失败静默降级为 0.5 分** | 失败计入 `windows_failed`、写入 `warnings`，指标置 `None` 并从计分中排除 | `test_parse_response_raises_when_no_json_present`、`test_window_failure_is_recorded_not_hidden`、`test_missing_semantic_is_excluded_not_defaulted` |
| 23 | 长转录截断丢中段 | 改为滑窗分块，完整覆盖 | `test_window_planning_covers_entire_discussion` |
| 24 | 丢弃的正是观点碰撞最密集处 | 同 #23；`_compress_transcript` 已移除 | `test_no_head_tail_truncation_api_remains` |
| 25 | `analyze_batch` 从未被调用 | 分窗分析即主路径 | `test_partial_failure_aggregates_only_successful_windows` |
| 26 | dashscope 分支引用不存在的 `qwen_model` | 统一由 `_resolve_model()` 解析 | `test_dashscope_model_resolution_does_not_crash` |
| 27 | 无超时/重试/限流/token 上限 | 全部补齐（`APIConfig`） | 代码 |
| 28 | `turn_taking_pattern` 算了不用 | 进入报告与诊断规则 | `test_end_to_end_over_real_fixture` + `_generate_diagnoses` |
| 29 | `consensus_quality` 算了不用 | 计入 `interaction_depth` 子项并驱动诊断 | `test_optional_semantic_metrics_are_respected` |

## 4. 工程与打包（#30–#46）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 30 | `whisper_device="auto"` 默认崩 | 规范化为 `None` | `test_whisper_device_normalisation`、`test_default_config_device_is_none_not_auto` |
| 31 | 4 个模块顶层 `warnings.filterwarnings("ignore")` | 全部移除 | `test_no_global_warning_filter_is_installed`（AST 检查） |
| 32 | `config.py` 导入即 `mkdir` | 改为显式 `get_paths(create=True)` | `test_import_creates_no_directories`、`test_get_paths_does_not_create_by_default` |
| 33 | `import` 即拉 `cv2` | 视觉依赖惰性导入 + PEP 562 惰性属性 | `test_importing_package_does_not_pull_heavy_dependencies` |
| 34 | 非 editable 安装后 `PROJECT_ROOT` 指向 site-packages，`.env` 读不到 | 回溯 `pyproject.toml` 定位，找不到则退回 cwd | `test_paths_do_not_point_into_site_packages`、`test_env_file_is_read_from_project_root` |
| 35 | 往 site-packages 里 `mkdir` | 同 #34 | 同上 |
| 36 | 0 个 `console_scripts` | 注册 `cla` 入口 | `test_console_script_is_declared`、`test_cli_end_to_end` |
| 37 | 顶层发布名为 `src` 的包 | 用 `package-dir` 映射，安装后顶层包名为 `collaborative_learning_analyzer` | `test_no_top_level_src_package_installed`、`test_installed_package_resolves_to_repo_src` |
| 38 | `load()` 丢失个体贡献度与模态明细 | 实现对称的 `from_dict` / `load` | `test_save_load_roundtrip_is_lossless`、`test_roundtrip_preserves_none_optional_metrics` |
| 39 | 依赖全 `>=` 无上限、不可复现 | 全部加上限；核心依赖降为空；按模态拆 extra | `test_core_dependencies_are_empty`、`test_all_extra_covers_every_component` |
| 40 | 文档的测试命令跑不起来 | 测试套件重写，命令可用 | 全量 `pytest` 通过 |
| 41 | 13 处未使用导入 | 全部清理，并加**自动化回归测试** | `test_no_unused_imports_in_source` |
| 42 | 3 个死方法 | 接入流水线或移除；加自动化检查 | `test_no_dead_module_level_helpers` |
| 43 | 声纹注册死代码（跨模态对齐唯一手段） | `align_speakers()` + `--align-profiles` | `test_load_transcript_*`、`test_align_*` |
| 44 | 0 tag / 0 release 却称 1.0.0 | 版本号回到与阶段相符的声明，并在 CHANGELOG 记录 | `test_development_status_matches_version` |
| 45 | Alpha 分类器与 1.0.0 自相矛盾 | 见 #44 | 同上 |
| 46 | 8 次提交集中在 36 分钟内 | 事实修正（见第 0 节），不作为缺陷项 | —— |

## 5. 测试体系（#47–#52）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 47 | 断言是同义反复，无回归防护 | 重写为可失败的边界断言 | 见下方变异测试结果 |
| 48 | 三个智能体零测试 | 补齐 audio / video / semantic 测试 | `tests/test_audio.py`、`tests/test_video.py`、`tests/test_semantic.py` |
| 49 | `test_core.py` 破坏 pytest 捕获 | 该文件移除；`pytest tests/` 可正常收集 | 全量运行不报错 |
| 50 | 用假数据分数充当质量证据 | 夹具改为真实媒体 + 显式局限声明 | `fixtures/README.md` |
| 51 | 自带 demo 用理想假数据也只有 83.4 分 | 演示脚本改为跑真实夹具，不再内置编造分数 | `examples/analyze_discussion.py` |
| 52 | 「100% 通过」实为只验自己算术 | 加入变异测试结论作为防护证据 | 见下方 |

## 6. 文档与诚信（#53–#68）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 53 | 许可证四种说法 | 全局统一 AGPL-3.0-or-later | `test_license_metadata_is_consistent` |
| 54 | LICENSE 是自写摘要，非协议全文 | 换为 GNU 官方 AGPL-3.0 全文（34,523 字节） | 同上（含长度断言） |
| 55 | 「伪造的评审记录」 | **撤回该指控**（见第 0 节）；`REVIEW_REPORT.md` 因内容误导性已移除 | `test_review_report_claims_match_reality` |
| 56 | 自评 4/5 同时承认核心无测试 | 随文件移除 | 同上 |
| 57 | 声称 Pydantic，实为 dataclasses | README 已改正 | README |
| 58 | 架构图写乘法、代码是加权和 | 架构图与公式全部重写 | `docs/SCORING_MODEL.md` |
| 59 | 健康分公式运算符优先级写错 | 重写为代码块形式 | 同上 |
| 60 | 快速开始引用的样例文件不存在且被 gitignore | 提供 `tools/make_fixtures.py` 生成真实媒体夹具 | `test_documented_sample_paths_exist_or_are_generated` |
| 61 | 项目结构漏列 7 个文件 | README 结构树与实际一致 | README |
| 62 | 隐私声明称本地处理，实际把转录发给云端 | README 明确区分「音视频不出本地」与「转录会发送给所配置的 LLM」，并给出本地部署方案 | README「隐私与数据边界」 |
| 63 | 「定期删除」无实现 | 删除该承诺，改为如实说明当前不提供数据生命周期管理 | 同上 |
| 64 | `python -m collaborative_learning_analyzer` 无 `__main__` | 补齐 `src/__main__.py` | `test_cli_module_is_runnable` |
| 65 | 复选框中放勾 | 路线图重写 | README |
| 66 | 0 star / 0 fork / 0 issue | 事实项，不修 | —— |
| 67 | 5 个月零更新、Stage 3 从未开始 | README「验证状态」如实标注未完成项 | README |
| 68 | 「一次提交却配两次提交记录」 | **撤回**（见第 0 节） | —— |

## 7. 验证方式

1. **单元与集成测试**：`pytest`，含边界、单调性、五级可达性、缺失模态、往返无损、打包与代码卫生。
2. **变异测试**：向计分逻辑注入缺陷，确认测试会变红 ——
   旧套件注入 3 处缺陷后仍 6/6 全绿；新套件对同类改动全部报错。
   命令与结果见 `docs/MUTATION_TESTING.md`。
3. **端到端**：在 `tools/make_fixtures.py` 生成的真实媒体夹具上跑通全流程
   （`tests/test_pipeline_e2e.py`）。
4. **未能验证的部分**：Whisper ASR、pyannote 说话人分离、YOLO 人体检测与视线估计的
   **精度**，因缺少真实课堂数据而未做评估。详见 README「验证状态」。
