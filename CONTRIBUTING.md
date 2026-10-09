# 贡献指南

感谢你考虑为本项目做出贡献。

## 报告问题

在 [Issues](https://github.com/firstxcst/collaborative-learning-analyzer/issues) 搜索是否已有类似问题，
然后创建新 Issue，包含：

- 清晰的标题与描述
- 复现步骤（含使用的命令、参数）
- 期望行为与实际行为
- 环境信息：Python 版本、操作系统、安装了哪些 extra（`pip list | grep -i -E "torch|opencv|whisper"`）
- **报告里输出的 `warnings` 字段内容** —— 本项目的降级信息都写在那里

## 提交代码

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 保持代码风格一致
4. **为新功能补测试**（见下）
5. 提交（`git commit -m 'Add some AmazingFeature'`）
6. 推送并开启 Pull Request

## 测试要求（重要）

本项目的测试必须**真的能失败**。旧版本的测试断言全是同义反复
（例如对已被 clamp 的值断言 `0 <= x <= 100`），注入三处严重缺陷后仍然全绿。
因此：

- 新增计分逻辑必须补**边界/单调性**断言，
  例如「沉默组的分数必须低于正常组」而不是「分数在 0 到 100 之间」；
- 涉及「不可用」语义的字段，断言必须是 `is None`，不要断言等于某个默认值；
- 提交前请自测：**故意把逻辑改坏，确认测试会变红**，再改回来。

```bash
pip install -e ".[all,dev]"

pytest                       # 全部测试
pytest -m "not slow"         # 跳过需要真实媒体夹具的慢测试
ruff check src tests
black src tests && isort src tests
```

需要真实媒体夹具时：

```bash
python tools/make_fixtures.py --out fixtures     # 仅 Windows
```

## 代码风格

- [Black](https://github.com/psf/black) 格式化，行宽 100
- [isort](https://github.com/PyCQA/isort) 排序导入（profile = black）
- [Ruff](https://github.com/astral-sh/ruff) 静态检查
- 遵循 PEP 8

## 数据与伦理

- **不要**向仓库提交任何真实学生的音视频、转录文本或可识别个人身份的数据，
  包括截图、日志与测试夹具。
- 测试请使用 `tools/make_fixtures.py` 生成的合成媒体，或自行构造的匿名数据。
- 涉及真实课堂数据的实验不得进入本仓库。

## 关于指标与结论

如果你要修改计分公式或权重，请在 PR 描述中说明：

1. 改动解决了什么具体问题（附复现方式）；
2. 依据是什么（文献、数据、或明确说明"这是工程默认值"）；
3. 新增/修改了哪些测试。

**不要**在没有真实课堂数据与一致性检验的情况下，在文档或 PR 中声称
「准确率提升」「效果更好」等结论。本项目已经因为把假数据跑出的分数当成质量证据
而付出过代价。

## 许可证

提交代码即表示你同意你的贡献在 **AGPL-3.0-or-later** 下授权（与本项目 LICENSE 一致）。
