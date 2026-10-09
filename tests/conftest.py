"""pytest 公共配置。

提供：
* 包导入回退（未执行 ``pip install -e .`` 时也能跑测试）；
* 真实媒体夹具的路径与跳过逻辑。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
FIXTURES_DIR = REPO_ROOT / "fixtures"
TRANSCRIPT_PATH = FIXTURES_DIR / "discussion.transcript.json"
AUDIO_PATH = FIXTURES_DIR / "discussion.wav"
VIDEO_PATH = FIXTURES_DIR / "discussion.mp4"

PACKAGE_NAME = "collaborative_learning_analyzer"


def _ensure_package_importable() -> None:
    """优先使用已安装的包；否则把 src/ 以包的形式动态注册进来。

    安装后的包名是 ``collaborative_learning_analyzer``（源码目录为 ``src/``），
    因此未安装时不能让 ``src`` 目录以 ``src`` 之名混入 sys.path。
    """
    if importlib.util.find_spec(PACKAGE_NAME) is not None:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        SRC_DIR / "__init__.py",
        submodule_search_locations=[str(SRC_DIR)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("无法加载 collaborative_learning_analyzer 包")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)


_ensure_package_importable()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def real_transcript_path() -> Path:
    if not TRANSCRIPT_PATH.is_file():
        pytest.skip(
            "缺少真实媒体夹具。请先运行：python tools/make_fixtures.py --out fixtures"
        )
    return TRANSCRIPT_PATH


@pytest.fixture(scope="session")
def real_audio_path() -> Path:
    if not AUDIO_PATH.is_file():
        pytest.skip(
            "缺少真实音频夹具。请先运行：python tools/make_fixtures.py --out fixtures"
        )
    return AUDIO_PATH
