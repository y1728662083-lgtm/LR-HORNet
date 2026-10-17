# Braindecode EEGNet 实验归档

本目录归档 4.4.1 中 `braindecode_eegnet` 的实际训练配置与关键代码快照，便于上传 Git、核查和复现实验。模型由 Braindecode 1.8.1 的公开 `EEGNet` 类直接构建；`baseline_model.py` 只向官方构造器传入数据集所需的信号元数据，没有增加模型包装层，也没有覆盖 EEGNet 的官方默认结构超参数。

## 实验范围

| 数据集 | Fold 数 | 输入元数据 |
| --- | ---: | --- |
| ds003626 | 10 | 128 × 1153，256 Hz |
| Bimodal inner speech | 4 | 64 × 359，512 Hz |
| Directional words — Russian | 12 | 34 × 250，250 Hz |
| Directional words — Spanish | 10 | 38 × 750，500 Hz |
| BCI2020 | 15 | 64 × 795，256 Hz |

Russian 与 Spanish 是两个独立数据集，分别取各自语言内部的公共通道，不跨语言对齐或混合。所有训练沿用 paper4 的带校准 LOSO 数据划分与相同训练引擎。

## 目录说明

- `configs/`：便于直接运行的有效配置，模型名已明确写为 `braindecode_eegnet`。
- `historical_configs/`：当时调用脚本读取的原始 YAML，未经修改，用于审计。
- `project_snapshot/`：实际训练入口、Braindecode 适配器、数据加载、划分、训练与评估代码快照。
- `historical_launchers/`：当时的全 Braindecode 模型目录启动脚本；它们会枚举完整官方模型表，仅用于追溯，不建议用来只重跑 EEGNet。
- `official_braindecode_1.8.1/`：当时 vendored Braindecode 版本号、注册表辅助代码与 EEGNet 官方实现。
- `official_preprocessing/bimodal/`：Bimodal 作者预处理脚本快照。
- `tests/`：当时的 Braindecode 基线升级/构造测试。
- `source_manifest.json` 与 `MANIFEST_SHA256.txt`：来源路径和文件校验信息。

`project_snapshot` 是审计快照，不是裁剪后的独立 paper4 仓库；训练时仍应放在完整 paper4 仓库中运行，以保留 factory、数据适配器及其全部依赖。

## 运行

在 AutoDL 的完整 paper4 环境中：

```bash
cd /path/to/this/EEGNet
PAPER4_ROOT=/path/to/LR-HORNet CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_all_datasets.sh
```

`run_all_datasets.sh` 复现原始四数据集批次。BCI2020 的历史 YAML 保留训练时绝对路径以维持哈希，便携运行应在仓库根目录通过 `run_bci2020_all_models_4gpu.sh` 生成本地运行配置。

默认复用已有 Bimodal derivatives。若需先执行归档的官方预处理链路：

```bash
PREPROCESS_BIMODAL=1 PAPER4_ROOT=/path/to/LR-HORNet \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_datasets.sh
```

历史结果未复制到本目录；它们仍位于 4.4.1 的 `dataset_result` 下对应 `braindecode_eegnet` 文件夹。
