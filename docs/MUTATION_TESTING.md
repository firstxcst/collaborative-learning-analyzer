# 变异测试结果

## 为什么做这件事

旧版本的测试断言是**同义反复**：例如

```python
assert 0 <= report.overall_health_score <= 100   # 代码本身已 clamp，永不失败
assert 0 <= total <= 1                            # 同上
```

因此「测试通过率 100%」并不代表任何东西。实测：向计分逻辑注入 3 处严重缺陷后，
旧套件仍然 **6/6 全绿**。

本文件记录新套件的**检出能力**验证结果：把旧缺陷逐个还原回去，确认测试会变红。

## 复现方式

```bash
python tools/mutation_check.py          # 全部变异
python tools/mutation_check.py --list   # 查看变异清单
```

脚本会：备份原文件 → 注入缺陷 → 运行 `pytest -q -x` → 还原文件。
只要有任一变异未被检出（测试仍全绿），脚本以非零码退出。

## 结果（2026-10-09）

基线（未注入缺陷）：

```
133 passed
```

注入 11 处缺陷后：

| 变异 | 注入的缺陷 | 结果 |
|------|-----------|------|
| `evenness_silent_returns_one` | 还原旧缺陷：全员沉默时均衡度返回 1.0 | ✅ 检出（1 failed） |
| `evenness_uses_absolute_score` | 均衡度改回「基于绝对分的标准差」思路 | ✅ 检出（1 failed） |
| `health_drops_activity_gate` | 去掉乘性活跃度门控 | ✅ 检出（1 failed） |
| `depth_computed_when_semantic_missing` | 语义不可用时仍计算交互深度 | ✅ 检出（1 failed） |
| `weights_not_renormalised` | 缺失维度不重新归一化权重 | ✅ 检出（1 failed） |
| `silent_parse_defaults_to_neutral` | 还原旧缺陷：解析失败静默降级为 0.5 | ✅ 检出（1 failed） |
| `load_drops_contributions` | 还原旧缺陷：`load()` 丢掉个体贡献度 | ✅ 检出（1 failed） |
| `whisper_device_accepts_auto` | 还原旧缺陷：把 `"auto"` 传给 whisper | ✅ 检出（1 failed） |
| `merge_assigns_all_overlaps` | 还原旧缺陷：同一句话重复分配给多个说话人 | ✅ 检出（1 failed） |
| `cohesion_hardcoded_default` | 还原旧缺陷：无视觉数据时返回 0.5 | ✅ 检出（1 failed） |
| `vision_imports_cv2_eagerly` | 还原旧缺陷：模块顶层 `import cv2` | ✅ 检出（1 error） |

```
结果：11/11 处缺陷全部被检出。测试具备回归防护能力。
```

## 覆盖不到的方面

变异测试只能验证「测试对**已写出的断言**是否有检出能力」，
它**不能**证明：

- 权重取值是否合理（这是标定问题，需要真实课堂数据）
- 语音 / 视觉链路的精度（需要真实课堂数据与人工标注基准）
- 指标的构念效度（需要与人工编码做一致性检验）

这三点在 README 的「验证状态」中列为**未验证**。

## 在 CI 中运行

变异测试耗时约 1 分钟（每个变异都要完整跑一遍套件），因此未加入默认 CI 流程。
建议在改动计分逻辑时手动运行一次。
