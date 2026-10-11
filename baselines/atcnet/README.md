# Braindecode ATCNet 实验归档

本目录归档 4.4.1 中 `braindecode_atcnet` 的实际训练配置与关键代码快照，便于上传 Git、核查和复现实验。模型由 Braindecode 1.8.1 的公开 `ATCNet` 类直接构建；`baseline_model.py` 只向官方构造器传入数据集所需的信号元数据，没有增加模型包装层，也没有覆盖 ATCNet 的官方默认结构超参数。

## 实验范围

| 数据集 | Fold 数 | 输入元数据 |
| --- | ---: | --- |
| ds003626 | 10 | 128 × 1153，256 Hz |
| Bimodal inner speech | 4 | 64 × 359，512 Hz |
| Directional words — Russian | 12 | 34 × 250，250 Hz |
| Directional words — Spanish | 10 | 38 × 750，500 Hz |

Russian 与 Spanish 是两个独立数据集，分别取各自语言内部的公共通道，不跨语言对齐或混合。所有训练沿用 paper4 的带校准 LOSO 数据划分与相同训练引擎。

## 配置说明

历史公用 YAML 的 `model.name` 是 `braindecode_eegnet` 占位值；实际运行时 `baseline_model.py --architecture ATCNet` 会在内存中将其替换成 `braindecode_atcnet`。因此：

- `historical_configs/` 保存原始 YAML，不做修改；
- `configs/` 将这一运行时替换结果显式写出，除此之外与原始 YAML 相同。

## 目录说明

- `project_snapshot/`：实际训练入口、Braindecode 适配器、数据加载、划分、训练与评估代码快照。
- `historical_launchers/`：当时的全 Braindecode 模型目录启动脚本；它们会枚举完整官方模型表，仅用于追溯，不建议用来只重跑 ATCNet。
- `official_braindecode_1.8.1/`：当时 vendored Braindecode 版本号、注册表辅助代码与 ATCNet 官方实现。
- `official_preprocessing/bimodal/`：Bimodal 作者预处理脚本快照。
- `tests/`：当时的 Braindecode 基线升级/构造测试。
- `source_manifest.json` 与 `MANIFEST_SHA256.txt`：来源路径和文件校验信息。

`project_snapshot` 是审计快照，不是裁剪后的独立 paper4 仓库；训练时仍应放在完整 paper4 仓库中运行，以保留 factory、数据适配器及其全部依赖。

## 运行

在 AutoDL 的完整 paper4 环境中：

```bash
cd /path/to/this/atcnet
PAPER4_ROOT=/path/to/LR-HORNet CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_all_datasets.sh
```

默认复用已有 Bimodal derivatives。若需先执行归档的官方预处理链路：

```bash
PREPROCESS_BIMODAL=1 PAPER4_ROOT=/path/to/LR-HORNet \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_datasets.sh
```

历史结果未复制到本目录；它们仍位于 4.4.1 的 `dataset_result` 下对应 `braindecode_atcnet` 文件夹。
