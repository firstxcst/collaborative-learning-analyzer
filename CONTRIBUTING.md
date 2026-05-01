# Contributing to Collaborative Learning Analyzer

感谢你考虑为本项目做出贡献！

## 如何贡献

### 报告问题

如果你发现了 bug 或有功能建议，请：
1. 在 [Issues](https://github.com/firstxcst/collaborative-learning-analyzer/issues) 页面搜索是否已有类似问题
2. 如果没有，创建新的 Issue，包含：
   - 清晰的标题和描述
   - 复现步骤（如果是 bug）
   - 预期行为和实际行为
   - 环境信息（Python 版本、操作系统等）

### 提交代码

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 确保代码风格一致
4. 为新功能添加测试
5. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
6. 推送到分支 (`git push origin feature/AmazingFeature`)
7. 开启 Pull Request

### 代码风格

- 使用 [Black](https://github.com/psf/black) 格式化代码
- 使用 [isort](https://pycqa.github.io/isort/) 排序导入
- 遵循 PEP 8 规范

### 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码格式化
black src/ tests/
isort src/ tests/
```

## 开发路线图

- [ ] Stage 1: MVP 数据流打通 ✅
- [ ] Stage 2: 多模态时序融合优化
- [ ] Stage 3: 真实课堂环境适配
- [ ] Stage 4: 教育专家评估
- [ ] Stage 5: Web 可视化界面

## 许可证

提交代码即表示你同意你的贡献将在 MIT 许可证下授权。
