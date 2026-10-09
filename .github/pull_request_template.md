## 这个 PR 做了什么 / What this PR does

<!-- 一到三句话说明改动与动机。 -->

## 类型 / Type

- [ ] Bug fix
- [ ] New feature
- [ ] Metric / scoring change（**必须**填下面「指标改动」一节）
- [ ] Docs / packaging / refactor

## 检查清单 / Checklist

- [ ] `pytest` 全绿
- [ ] 新增逻辑带了**会失败的**断言（不是同义反复）
- [ ] 涉及计分逻辑时，已运行 `python tools/mutation_check.py` 并确认全部检出
- [ ] README / CHANGELOG 已同步（行为、字段、命令有变化时必须更新）
- [ ] 没有提交真实学生的未脱敏音视频、转录或个人信息
- [ ] 没有把未验证的能力写成已验证（见 README「验证状态」）

## 指标改动 / Metric changes

<!-- 仅当改动计分公式、权重或阈值时填写 -->

- 解决的问题：
- 依据（文献 / 数据 / 明确标注为工程判断）：
- 验证方式（哪条断言、哪份数据）：

## 行为变化 / Behaviour changes

<!-- 破坏性变更请写明：字段重命名、默认值变化、输出结构变化。 -->
