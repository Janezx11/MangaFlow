# RELEASE CHECKLIST — v0.1.0 上架前核对清单

> 用途：发布 v0.1.0 到 GitHub 前逐项核对。每项都有可执行的命令/动作。
> 状态说明：本机（G:\trainslate）已完成的事已标 ✅；需要"有网/真实环境"的项标 🔲 待你执行。

---

## 0. 已在本会话完成 ✅

- [x] git 仓库初始化 + 干净历史（多轮提交，无大文件/密钥）
- [x] `.gitignore` 完整（含修复 `models/` 误伤源码包 bug，`/models/` 已锚定）
- [x] README（简介/快速开始/安装/配置/模型/已知限制）
- [x] LICENSE（MIT）
- [x] `pyproject.toml` 依赖补全（core 15 + 10 extras）
- [x] `config.example.yaml` 与 config.yaml 键同步
- [x] 死代码清理（glossary/manager+matcher、defaults.yaml、bubble_region、merged_ocr_block、utils/image、typography v1 标注）
- [x] 关键 bug 修复：翻译失败块标记、渲染继承原文件名、模型库启用生效、训练导出路径嵌套、数据集路径错位
- [x] 翻译提速：文本去重
- [x] GUI 逐次修改后**已由你在真实环境跑通**（设置页填 key/拉模型、模型库、翻译流程）

---

## 1. 🔴 发布前必做（需要「有网 + 真实环境」，约 1~2 小时）

- [ ] **clone 自检（最重要）**：在一台干净/有网机器上
  ```bash
  git clone <你的仓库URL> comic-translator
  cd comic-translator
  python -m venv .venv && .\.venv\Scripts\activate   # 或 uv venv
  pip install -e ".[all]"
  python -c "import comic_translator; from comic_translator.models.bubble import DialogueBlock; print('import OK')"
  pytest
  ```
  预期：import 不报错、测试全绿（本机沙箱下 54 个 `tmp_path` 权限错误在正常环境应消失）。
- [ ] **ruff + mypy 全绿**（本项目从未在真实环境跑过这两个，是唯一"未知数"）：
  ```bash
  ruff check src tests
  mypy src/comic_translator
  ```
  若报错较多：先修 `pyproject.toml` 的 `[tool.ruff] select` 收敛规则或修复代码。
- [ ] **旧 API key 轮换**：在商汤 Sensenova 控制台吊销旧 key（`sk-jt5dLm...`），生成新 key，用 GUI 设置页填入（config.yaml 已 gitignore，安全）。
- [ ] **GUI 完整走一遍**（你已逐次跑过 ✅，发布前再确认一次全流程）：填 key → 选图 → 翻译 → 渲染 → 输出图片名与原图对应。

---

## 2. 🟡 强烈建议（发布时一起做，各 1~2 小时）

- [ ] **README 加翻译前后对比图**（截 2~3 张：原图 / 渲染后 / 失败标记示例）
- [ ] **GitHub Actions CI**：`.github/workflows/ci.yml`，跑 `ruff + mypy + pytest + 构建 wheel`
- [ ] **`download-models` 命令**（PLAN §3.7）：`python -m comic_translator download-models` 从 HuggingFace 拉 yolo/ctbd/lama

---

## 3. 🟢 v1.1 再补（可选，不阻塞 v1）

- [ ] 微型 demo 数据集 + 导入脚本（PLAN §3.9，让 ML 标注/训练可点通）
- [ ] 重构 `gui_pipeline_service` 单体（1801 行）
- [ ] 多语言 OCR（ja/zh）、CBZ/PDF 导出、交互式翻译编辑

---

## 4. 发布动作（GitHub 侧）

1. 创建 GitHub 仓库（public，仓库名建议 `comic-translator`）
2. push：
   ```bash
   git remote add origin <url>
   git branch -M main
   git push -u origin main
   ```
3. 仓库元数据：description（一句话）、topics（`comic-translation` `ocr` `llm` `gradio` `yolo` `image-inpainting` 等）
4. 打 tag 发布 v0.1.0：GitHub Release，标题写清功能，附翻译前后对比图，标注"已知限制"
5. 发布后第一件事：看有没有人 clone 后报安装问题，优先修

---

## 5. 一句话提醒

**先跑通 1 里的 clone 自检 + ruff/mypy，再 push 公开。** 只要这两项绿了，剩下的都是锦上添花。
