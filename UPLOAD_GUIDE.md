# GitHub 上传指引 — H3-Adaptive-Generation

本文件说明如何将 `CodeDataAvailability/` 内容上传至：
https://github.com/phenshine/H3-Adaptive-Generation

---

## 方法一：Git 命令行（推荐）

```bash
# 1. 克隆仓库（使用您的 GitHub Token）
git clone https://github.com/phenshine/H3-Adaptive-Generation.git
cd H3-Adaptive-Generation

# 2. 将 CodeDataAvailability/ 下的所有文件复制到仓库根目录
#    （在 Windows 上用资源管理器复制，或用以下命令）
xcopy /E /I /Y "<本地路径>\CodeDataAvailability\*" "."

# 3. 提交并推送
git add .
git commit -m "feat: add CD-MCAR implementation, data, and documentation

- src/cd_mcar.py: core CD-MCAR algorithm
- src/slmm.py: SLMM boundary encoding
- src/h3_utils.py: h3-py v4 utility wrappers
- experiments/: all pipeline scripts (Steps 2-8)
- data/sample/: derived vector datasets (CC BY 4.0)
- docs/: API reference and PostgreSQL schema
- README.md, requirements.txt, LICENSE"

git push origin main
```

---

## 方法二：GitHub 网页上传

1. 访问 https://github.com/phenshine/H3-Adaptive-Generation
2. 点击 **"Add file"** → **"Upload files"**
3. 拖入 `CodeDataAvailability/` 下的所有文件和文件夹
4. 填写 Commit message，点击 **"Commit changes"**

> **注意**：网页上传不支持超过 100 个文件，建议分批上传或使用 Git 命令行。

---

## 方法三：GitHub Desktop

1. 下载安装 [GitHub Desktop](https://desktop.github.com/)
2. Clone `phenshine/H3-Adaptive-Generation`
3. 在 Finder/资源管理器中将 `CodeDataAvailability/` 内容复制到仓库目录
4. 在 GitHub Desktop 中 Commit & Push

---

## 上传后验证清单

- [ ] README.md 正确渲染（含徽章和表格）
- [ ] `src/cd_mcar.py` 可在 GitHub 网页预览
- [ ] `data/sample/detections.geojson` 在 GitHub 自动渲染地图
- [ ] LICENSE 文件显示为 MIT License
- [ ] requirements.txt 格式正确
- [ ] 在仓库 Settings 中确认 License 类型为 MIT

---

## 论文中的引用格式

完成上传后，在论文 Data/Code Availability 章节填写：

**Data Availability:**
The derived vector datasets, including `detections.geojson`, `adaptive_tree_leaves.csv`,
`boundary_encoding.json`, and `h3_adaptive_grid.geojson`, are publicly available on GitHub at:
https://github.com/phenshine/H3-Adaptive-Generation

**Code Availability:**
The CD-MCAR algorithm implementation and all experimental scripts are publicly available at:
https://github.com/phenshine/H3-Adaptive-Generation
The code is released under the MIT License.
