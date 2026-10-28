# 数据放置说明（本包**不含任何数据**）

本准备包只包含代码、配置副本与已完成实验的 JSON 指标，不包含原始数据、模型权重、
日志或诊断 npz。复现前请自行获取数据并按下列目标路径放置；本包不会自动下载任何数据。
`README.md` 的 “Required data and layout” 一节与 `DATA_SOURCES.md` 是权威说明，如与
本文件不一致以它们为准。

## 热场数据（原作者数据，第三方，不随本包分发）

- 来源、下载地址与 sha256 见包内 `DATA_SOURCES.md`（由 Codex 提供；请核对下载文件的
  sha256 与其中记录一致）。
- 放置目标：包根目录下的 `data/thermal/HeatTransferData.npy`（或运行
  `scripts/exp_thermal.py` 时用 `--data <你的路径>/HeatTransferData.npy` 显式指定）。
  形状/列序约定见 `README.md` 与 `src/core/thermal_io.py`。

## 声学数据（作者单独的声学数据包，不随本包分发）

- 将作者声学数据包中的 `case_<id>.npz` 放到包根目录下的 `data/acoustic/`，共 21 个
  case，文件名必须与 `configs/acoustic_cases.json` 中每个 case 的 `file` 字段一致
  （`data/acoustic/case_{f}_{Ma}.npz`）；不要修改共享配置。
- 未放置数据时，本包只提供代码与 JSON 指标，不能声称已完成整场声学评估复现。

## 不包含的内容

- 模型权重（`*.pth`/`*.pt`）、训练/评测日志、诊断 npz 均不在本包。
- 未完成实验（FDTD 压力收敛）的结果不在本包；状态见 `PENDING_STATUS.json`。
- new_queries 的已完成 JSON 在本包内，但其参考是插值参考诊断（interpolated-reference
  diagnostic），不是独立 FEM 解；PENDING_STATUS.json 中有对应范围说明。
