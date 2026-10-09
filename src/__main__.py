"""``python -m collaborative_learning_analyzer`` 入口。

修复的审计问题：
* 旧 ``src/__main__.py`` 内嵌了一个硬编码假数据的“演示模式”，既不是真 CLI，
  也没有被注册为命令；现已改为调用真正的 CLI（``cli.main``）。
* 旧 ``collaborative_learning_analyzer/__init__.py`` 的文档声称
  ``python -m collaborative_learning_analyzer`` 可用，但该包下并没有 ``__main__.py``。
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
