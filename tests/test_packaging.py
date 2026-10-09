"""工程与打包测试。

覆盖的审计问题：#30 #31 #32 #33 #34 #35 #36 #37 #41 #42 #44 #45 #64
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
PACKAGE = "collaborative_learning_analyzer"


def run_python(code: str) -> subprocess.CompletedProcess:
    """在全新解释器中执行代码，用于隔离导入期副作用。"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )


# --------------------------------------------------------------- 惰性导入

def test_importing_package_does_not_pull_heavy_dependencies():
    """修复 #33：旧 ``src/__init__.py`` 导入即拉 cv2，缺 opencv 时连 import 都失败。"""
    code = (
        f"import {PACKAGE} as m;"
        "import sys;"
        "heavy=[n for n in ('cv2','torch','whisper','ultralytics','soundfile','numpy')"
        " if n in sys.modules];"
        "print(','.join(heavy))"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"导入包时不应加载重型依赖，实际加载了 {result.stdout.strip()}"


def test_core_objects_import_without_vision_stack():
    """核心对象（含 FusionEngine / AudioAgent）必须能在无 opencv 环境下使用。"""
    code = (
        f"from {PACKAGE} import AudioAgent, FusionEngine, GroupCollaborationReport;"
        "print(FusionEngine.__name__, AudioAgent.__name__, GroupCollaborationReport.__name__)"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "FusionEngine" in result.stdout


def test_import_creates_no_directories():
    """修复 #32/#35：旧 config.py 在 import 时直接 mkdir 三个目录（安装后还会写进 site-packages）。"""
    code = (
        "import tempfile, os, pathlib, sys;"
        "tmp = tempfile.mkdtemp();"
        "os.chdir(tmp);"
        f"import {PACKAGE};"
        "print(sorted(p.name for p in pathlib.Path(tmp).iterdir()))"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"导入包时不应创建任何目录，实际创建了 {result.stdout.strip()}"


def test_no_top_level_src_package_installed():
    """修复 #37：安装后的顶层包名必须是 collaborative_learning_analyzer，而不是 src。"""
    code = "import sysconfig, pathlib; sp=pathlib.Path(sysconfig.get_paths()['purelib']); print((sp/'src').exists())"
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().lower() == "false"


def test_installed_package_resolves_to_repo_src():
    code = f"import {PACKAGE} as m; import pathlib; print(pathlib.Path(m.__file__).parent.resolve())"
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == SRC_DIR.resolve()


# --------------------------------------------------------------- 路径解析

def test_paths_do_not_point_into_site_packages():
    """修复 #34：旧实现在非 editable 安装后会把 PROJECT_ROOT 解析成 site-packages，
    导致用户放在项目根的 .env 永远读不到，还会往依赖目录里 mkdir。"""
    from collaborative_learning_analyzer.config import get_paths

    paths = get_paths()
    assert "site-packages" not in str(paths.project_root)
    assert "site-packages" not in str(paths.output)


def test_paths_respect_env_override(monkeypatch, tmp_path):
    from collaborative_learning_analyzer.config import get_paths

    monkeypatch.setenv("CLA_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CLA_OUTPUT_DIR", raising=False)
    paths = get_paths()
    assert paths.project_root == tmp_path.resolve()
    assert paths.output == tmp_path.resolve() / "results"


def test_get_paths_does_not_create_by_default(tmp_path, monkeypatch):
    from collaborative_learning_analyzer.config import get_paths

    monkeypatch.setenv("CLA_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CLA_DATA_DIR", raising=False)
    monkeypatch.delenv("CLA_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("CLA_MODELS_DIR", raising=False)

    paths = get_paths(create=False)
    assert not paths.output.exists()
    assert not paths.data.exists()

    get_paths(create=True)
    assert paths.output.exists() and paths.data.exists() and paths.models.exists()


def test_env_file_is_read_from_project_root(tmp_path, monkeypatch):
    from collaborative_learning_analyzer.config import load_api_config, reset_config

    monkeypatch.setenv("CLA_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    reset_config()
    try:
        assert load_api_config().openai_api_key == "from-dotenv"
    finally:
        reset_config()


# --------------------------------------------------------------- CLI / 入口

def test_cli_module_is_runnable():
    """修复 #64：旧文档声称可用 ``python -m collaborative_learning_analyzer``，但缺少 __main__。"""
    result = run_python(f"import runpy; runpy.run_module('{PACKAGE}.__main__', run_name='__not_main__')")
    assert result.returncode == 0, result.stderr


def test_console_script_is_declared():
    """修复 #36：旧仓库一个 console_scripts 都没有，装完没有命令可用。"""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"].get("scripts", {})
    assert scripts.get("cla") == f"{PACKAGE}.cli:main"


def test_cli_paths_command_outputs_json():
    result = run_python(f"from {PACKAGE}.cli import main; main(['paths'])")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {"project_root", "data", "output", "models"}


def test_cli_analyze_requires_input():
    result = run_python(f"from {PACKAGE}.cli import main; raise SystemExit(main(['analyze']))")
    assert result.returncode == 2


# --------------------------------------------------------------- 发布元数据

def test_version_consistency():
    import tomllib

    import collaborative_learning_analyzer as package

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert package.__version__ == data["project"]["version"]


def test_development_status_matches_version():
    """修复 #45：旧 pyproject 一边声明 1.0.0 正式版，一边标记 Alpha，自相矛盾。"""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = " ".join(data["project"]["classifiers"])
    version = data["project"]["version"]
    if "Development Status :: 3 - Alpha" in classifiers:
        assert version.startswith("0.") or "+" in version or version.endswith(".0"), (
            "Alpha 阶段不应声称稳定正式版"
        )
    assert "License ::" not in classifiers, "PEP 639 要求移除 license 分类器（否则构建会失败）"


def test_core_dependencies_are_empty():
    """核心依赖为空：装本包不应强迫用户下载 torch。"""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == []
    extras = data["project"]["optional-dependencies"]
    for name in ("audio", "vision", "llm", "qwen", "voiceprint", "diarization", "dev", "all"):
        assert name in extras, f"缺少 extra: {name}"


def test_all_extra_covers_every_component():
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    all_extra = " ".join(data["project"]["optional-dependencies"]["all"])
    for name in ("audio", "vision", "llm", "voiceprint"):
        assert name in all_extra


def test_license_metadata_is_consistent():
    """修复 #53：旧仓库对许可证有四种说法（README 徽章 MIT、正文 AGPL、
    LICENSE 是自写摘要、CONTRIBUTING 又是 MIT）。这里要求全局一致。"""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = data["project"]["license"]
    assert declared == "AGPL-3.0-or-later"

    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "完整许可证文本请参阅" not in license_text, "LICENSE 必须是协议全文，不能是自写摘要"
    assert len(license_text) > 30000, "AGPL-3.0 全文约 34KB，当前文件明显不是全文"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "License-MIT" not in readme, "README 不应再声称 MIT"
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "MIT" not in contributing, "CONTRIBUTING 不应再声称 MIT"


def test_py_typed_is_shipped():
    """PEP 561 标记文件必须存在；旧文件里塞了注释，不是合法的空标记。"""
    marker = SRC_DIR / "py.typed"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == ""


# --------------------------------------------------------------- 代码卫生

def _unused_imports(path: Path) -> list[str]:
    """用 AST + 文本出现次数判断未使用的导入（跳过 __init__ 的再导出）。"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    imported: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(((alias.asname or alias.name).split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name != "*":
                    imported.append((alias.asname or alias.name, node.lineno))

    unused = []
    for name, lineno in imported:
        if name == "annotations":
            continue
        body = [line for index, line in enumerate(lines, 1) if index != lineno]
        # 名字出现在别的行，或出现在同一行的别名/括号导入中，都算已使用
        if not any(name in line for line in body):
            unused.append(f"{path.name}:{lineno} {name}")
    return unused


def test_no_unused_imports_in_source():
    """修复 #41：旧源码有 13 处未使用的导入。"""
    offenders: list[str] = []
    for path in sorted(SRC_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue  # 再导出模块，符号用于对外暴露
        offenders.extend(_unused_imports(path))
    assert not offenders, "存在未使用的导入：" + ", ".join(offenders)


def test_no_dead_module_level_helpers():
    """修复 #42：旧源码里 register_speaker / load_speaker_profiles / analyze_batch
    定义完从未被调用。这里检查定义是否在**仓库范围内**至少被引用一次
    （src / tests / examples / tools 都算使用方）。"""
    scan_dirs = [SRC_DIR, REPO_ROOT / "tests", REPO_ROOT / "examples", REPO_ROOT / "tools"]
    sources = {
        p: p.read_text(encoding="utf-8")
        for directory in scan_dirs
        if directory.is_dir()
        for p in sorted(directory.rglob("*.py"))
    }
    src_sources = {p: s for p, s in sources.items() if p.parent == SRC_DIR}
    joined = "\n".join(sources.values())

    dead: list[str] = []
    for path, source in src_sources.items():
        if path.name in {"__init__.py", "__main__.py", "cli.py"}:
            continue
        tree = ast.parse(source)
        classes = {n for n in tree.body if isinstance(n, ast.ClassDef)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("__") or node.name.startswith("test_"):
                continue
            # 跳过会由框架/外部调用的入口
            if node.name in {"main", "analyze"}:
                continue
            if joined.count(node.name) <= 1:
                is_method = any(node in cls.body for cls in classes)
                dead.append(
                    f"{path.name}:{node.lineno} {'method' if is_method else 'function'} {node.name}"
                )
    assert not dead, "存在从未被引用的定义：" + ", ".join(dead)


def test_review_report_claims_match_reality():
    """修复 #55/#67：仓库不得再保留把自评包装成“社区评审”的文档。"""
    for name in ("REVIEW_REPORT.md", "PROJECT_SUMMARY.md"):
        assert not (REPO_ROOT / name).exists(), f"{name} 应已移除（内容与事实不符/含过期声明）"
    assert (REPO_ROOT / "CHANGELOG.md").is_file()


def test_documented_sample_paths_exist_or_are_generated():
    """修复 #60：旧 README 的快速开始引用 data/sample.wav，而该文件不存在且被 gitignore，
    任何人 clone 下来都跑不出结果。现在改为由脚本生成夹具。"""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "data/sample.wav" not in readme
    assert "tools/make_fixtures.py" in readme
    assert (REPO_ROOT / "tools" / "make_fixtures.py").is_file()
    assert (REPO_ROOT / "fixtures" / "README.md").is_file()
