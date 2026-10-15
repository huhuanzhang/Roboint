# 上传 GitHub

目标仓库：[huhuanzhang/Roboint](https://github.com/huhuanzhang/Roboint)。本地通过 HTTPS `origin` 同步到 `main` 分支，保留原有 Git 历史。

## 本次准备

- 保留现有源码、配置、测试和两次历史提交。
- `.gitignore` 排除根目录 `data/`、`outputs/`、权重、缓存、`.env`、备份和 `tmp.md`；文件仍保留在本机。
- 本机路径改为环境变量或相邻 checkout，见根目录 `.env.example`。
- 普通测试不要求本机存在原测试机的参考文件；wire parity 需显式开启。
- GitHub Actions 配置 Python 3.10 / 3.12 的 CPU 测试、Mock 全链路和构建。
- 2026 年 9 月真机验证记录移至 `VALIDATION_HISTORY.md`，与当前安装说明分开维护。

## 本地检查

在仓库根目录、已安装开发依赖的环境中执行：

```bash
git status --short
git diff --check
python -m ruff check --select E9,F63,F7,F82 robo_int scripts tests
python -m pytest tests/ -q
```

数据和权重不随源码上传。普通 CPU 环境会跳过需要 LeRobot、Piper SDK、真实数据、在线服务或 GPU 的验证。
`.env` 不自动加载；真机运行前按 `.env.example` 填写并导出环境变量。

## 本地验证结果（2026-10-15）

在不包含实验数据的源码副本中，使用独立 Python 3.12.13 / PyTorch 2.14.1+cpu 环境验证：

- 完整回归：124 passed，16 skipped（外部数据、LeRobot/Piper 依赖、参考服务或 GPU 等前置条件未满足）。
- Mock 脚本全链路：采集 5 条、清洗、20 步 SFT、checkpoint 加载评测，全部正常退出；这是流程验证，不是算法效果评估。
- Python 语法及未定义名称检查通过；源码发行包与 wheel 构建通过，发行包包含配置、脚本、测试和文档。
- 当前文件和两次历史提交的常见凭据模式扫描未发现命中。
- 单独验证本机 LeRobot 数据集依赖不完整、参考文件不存在时，相关可选测试会跳过。

本次未重新验证真机运动、GPU 模型训练或真实 LeRobot 数据集转换。

## 后续更新

在本地 `main` 分支检查并提交更新：

```bash
git add -A
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare Robo_int for GitHub"
git push origin main
```

执行前可用 `git ls-files --cached` 检查暂存文件列表。GitHub Actions 的运行结果见仓库的 Actions 页面。
同级目录的 `robo_int_github_ready_20261015.zip` 是不含 `.git` 的源码快照；保留现有 Git 历史请从原项目目录推送。

## 项目许可

`pyproject.toml` 当前声明 `Proprietary`，本次保留该设置，未添加其他 LICENSE。
外部模型、数据、硬件 SDK 与引用实现的来源说明保留在相应模块文档中。
