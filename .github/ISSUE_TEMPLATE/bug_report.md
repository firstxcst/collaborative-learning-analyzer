---
name: Bug report
about: Something is broken or produces a wrong result
title: "[Bug] "
labels: bug
---

<!--
请先确认 / Before filing:
1. 已阅读 README 的「验证状态」一节 —— 语音/视觉的精度问题属于“未验证”，
   不属于 bug，但欢迎附上真实数据与复现方式。
   Read the "Verification status" section first: unimplemented accuracy evaluation is
   documented as unverified, not a bug — but real data + reproduction is very welcome.
2. 请勿在 issue 中上传任何真实学生的音视频、转录文本或可识别个人身份的信息。
   Never upload real student media, transcripts, or personally identifiable data.
-->

## 环境 / Environment

- OS:
- Python (`python -V`):
- Installed extras (`pip list | grep -iE "torch|opencv|whisper|ultralytics"`):
- `cla --version`:

## 复现 / Reproduction

```
# exact commands
```

## 实际行为 / Actual behaviour

```
# paste the output, including the `warnings` field if a report was produced
```

## 期望行为 / Expected behaviour

## 是否出现在 `warnings` 里 / Does it appear in `warnings`?

<!--
本项目所有降级与失败都会写进报告的 warnings 字段。如果你看到的是
“某个维度显示为不可用”，那通常是设计如此（缺失模态不参与计分，不会被编造）。
-->

## 使用的数据 / Data used

<!--
请说明数据来源：合成夹具 / 自录 / 已脱敏的真实课堂数据。
不要附上未脱敏的真实数据。
-->
