# HDF5 Targets 按 TargetConfig 索引存储方案

## 1. 概述

### 1.1 目标
将 HDF5 targets 存储结构从按 `target_type` 分组改为按 `TargetConfig` 索引组织（`/targets/0`, `/targets/1`, `/targets/2`...），简化数据结构，所有元数据移至 manifest.json。

### 1.2 现状分析

**当前 HDF5 结构**：
```
/targets/
    /{target_type}/data    - 按 target_type 分组存储
```

**问题**：
- 同一 target_type 的多个 targets 被堆叠存储为 `(num_seqs, target_length, num_targets_of_type)`
- 不同 TargetConfig 的 target_length 必须相同
- 无法支持不同 TargetConfig 有不同 target_length 的场景

**迁移目标**：
- 所有 targets 按 TargetConfig 索引单独存储
- 每个 target 的数据 shape 为 `(num_seqs, target_length_i)` 2D tensor
- 不再需要 target_type 分组

### 1.3 目标结构

```
/targets/
    /0              - float32 array (num_seqs, target_length_0)
    /1              - float32 array (num_seqs, target_length_1)
    /2              - float32 array (num_seqs, target_length_2)
```

**简化要点**：
1. 每个 target 直接用 index 作为 dataset 名：`/targets/0`, `/targets/1`, `/targets/2`
2. data 的 shape 是 `(num_seqs, target_length)` 2D tensor
3. 所有元数据存储在 manifest.json 中，HDF5 只存储数据

---

## 2. 详细设计

### 2.1 HDF5Writer.write_split 修改

**文件**: `src/bernese/data/backends/hdf5.py`

**接口变更**：
```python
def write_split(
    self,
    split: str,
    chrom_names: list[str],
    starts: list[int],
    ends: list[int],
    targets: list[np.ndarray] | None = None,  # 改为 list[np.ndarray]
    target_configs: list[TargetConfig] | None = None,  # 新增：TargetConfig 列表
    chunk_size: int = 1024,
) -> None:
```

**写入逻辑**：
```python
# 写入 targets group
if targets is not None and target_configs is not None:
    targets_grp = f.create_group("targets")
    
    for i, (target_data, config) in enumerate(zip(targets, target_configs)):
        # 创建 /targets/{i} dataset（直接作为 dataset，不是 group）
        target_length = target_data.shape[1]
        
        targets_grp.create_dataset(
            str(i),
            data=target_data.astype(np.float32),
            chunks=(chunk_size, target_length),
            compression="gzip",
            compression_opts=4,
        )
```

**排序逻辑**：
- `targets` 列表按 `sorted_idx` 排序后重新组织
- 每个 target_data 单独排序

### 2.2 HDF5Backend.get_targets 修改

**文件**: `src/bernese/data/backends/hdf5.py`

**新签名**：
```python
def get_targets(
    self,
    split: str,
    indices: np.ndarray | list[int] | slice | None = None,
    target_index: int | None = None,  # 新增：指定读取哪个 target
) -> torch.Tensor | list[torch.Tensor]:
    """Load targets for a split.
    
    Args:
        split: Dataset split name (train/valid/test)
        indices: Specific indices to load, or None for all
        target_index: Specific target index to load, or None for all targets
        
    Returns:
        If target_index is specified: Tensor of shape (batch, target_length)
        If target_index is None: List of tensors, one per TargetConfig
    """
```

**读取逻辑**（新格式，仅 `/targets/{index}`）：
```python
with h5py.File(self._split_files[split].filename, "r") as f:
    if "targets" not in f:
        raise KeyError(f"No targets found in {split_info.split_file}")
    
    targets_grp = f["targets"]
    
    if target_index is not None:
        # 读取指定 target：/targets/{target_index}
        target_path = str(target_index)
        if target_path not in targets_grp:
            raise KeyError(f"Target {target_index} not found in {split}")
        
        targets_ds = targets_grp[target_path]
        if indices is not None:
            data = targets_ds[indices]
        else:
            data = targets_ds[:]
        return torch.from_numpy(data.astype(np.float32))
    else:
        # 读取所有 targets
        results = []
        num_targets = len(self._metadata.targets)
        for i in range(num_targets):
            target_path = str(i)
            if target_path not in targets_grp:
                raise KeyError(f"Target {i} not found in {split}")
            
            targets_ds = targets_grp[target_path]
            if indices is not None:
                data = targets_ds[indices]
            else:
                data = targets_ds[:]
            results.append(torch.from_numpy(data.astype(np.float32)))
        return results
```

### 2.3 DatasetMetadata 扩展

**文件**: `src/bernese/data/backends/base.py`

**新增字段**：
```python
@dataclass
class DatasetMetadata:
    # ... 现有字段 ...
    
    # 新增：target_lengths 字典，按索引存储每个 target 的长度
    target_lengths: dict[int, int] = field(default_factory=dict)
```

**from_manifest 扩展**：
```python
# 在 from_manifest 中解析 target_lengths
if "targets" in data and "lengths" in data["targets"]:
    for idx, length in data["targets"]["lengths"].items():
        target_lengths[int(idx)] = length
```

**to_dict 扩展**：
```python
# 在 to_dict 中输出 target_lengths
"targets": {
    "num_targets": self.num_targets,
    "lengths": self.target_lengths,  # 每个 target 的长度
    "pool_width": self.pool_width,
    "diagonal_offset": self.diagonal_offset,
    "info": [],
}
```

### 2.4 GenomicDataset 修改

**文件**: `src/bernese/data/dataset.py`

**`__getitem__` 返回格式变更**：

当前：
```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 返回 (sequence, targets) 其中 targets shape: (target_length, num_targets)
```

新：
```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # 返回 (sequence, targets_list) 其中 targets_list[i] shape: (target_length_i,)
```

**`get_batch` 返回格式变更**：

当前：
```python
def get_batch(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 返回 (sequences, targets) 其中 targets shape: (batch, target_length, num_targets)
```

新：
```python
def get_batch(self, start: int, end: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # 返回 (sequences, targets_list) 其中 targets_list[i] shape: (batch, target_length_i)
```

### 2.5 Trainer 修改

**文件**: `src/bernese/training/trainer.py`

**Loss 计算适配**：

当前期望：`y shape: (batch, target_length, num_targets)`

新格式：`y 是 list[Tensor]`，每个 Tensor shape: `(batch, target_length_i)`

**展平拼接方案**：
```python
def _prepare_targets(self, targets_list: list[torch.Tensor]) -> torch.Tensor:
    """将 targets list 展平拼接为单一 tensor."""
    # 沿最后一维拼接
    return torch.cat(targets_list, dim=-1)  # (batch, target_length_total)
```

---

## 3. manifest.json 格式

### 3.1 格式

```json
{
  "version": "3.0",
  "name": "dataset_name",
  "created": "2026-03-27T00:00:00Z",
  "genome": {
    "name": "genome_name"
  },
  "sequences": {
    "seq_length": 131072,
    "seq_depth": 4,
    "splits": {
      "train": {"num_seqs": 1000, "split_file": "train.h5"},
      "valid": {"num_seqs": 100, "split_file": "valid.h5"},
      "test": {"num_seqs": 200, "split_file": "test.h5"}
    }
  },
  "targets": {
    "num_targets": 3,
    "lengths": {
      "0": 896,
      "1": 1024,
      "2": 512
    },
    "pool_width": 128,
    "diagonal_offset": 2,
    "info": [
      {
        "name": "MicroC_rep1",
        "target_type": "hic",
        "clip": 2.0
      },
      {
        "name": "MicroC_rep2",
        "target_type": "hic",
        "clip": 2.0
      },
      {
        "name": "ATAC-seq",
        "target_type": "bigwig",
        "clip": null
      }
    ]
  },
  "statistics": {}
}
```

**说明**：
- `lengths` 字典存储每个 target index 对应的 target_length
- `info` 数组存储每个 target 的详细配置（name, target_type, clip 等）
- HDF5 中不再存储任何属性，所有元数据集中在 manifest.json

---

## 4. 实现步骤

### 步骤 1: 修改 DatasetMetadata
- [ ] 在 `DatasetMetadata` 添加 `target_lengths` 字段
- [ ] 更新 `from_manifest` 和 `to_dict` 方法

### 步骤 2: 修改 HDF5Writer.write_split
- [ ] 更新方法签名接收 `targets: list[np.ndarray]` 和 `target_configs: list[TargetConfig]`
- [ ] 实现按索引存储逻辑（`/targets/{i}` 直接作为 dataset）
- [ ] 更新排序逻辑

### 步骤 3: 修改 HDF5Backend.get_targets
- [ ] 添加 `target_index` 参数
- [ ] 实现按索引读取逻辑（`/targets/{i}`）

### 步骤 4: 修改 preparation.py
- [ ] 更新 `_extract_targets_and_write` 方法
- [ ] 传递 targets list 和 target_configs 到 writer

### 步骤 5: 修改 GenomicDataset
- [ ] 更新 `__getitem__` 返回 list[Tensor]
- [ ] 更新 `get_batch` 返回 list[Tensor]
- [ ] 更新缓存逻辑

### 步骤 6: 修改 Trainer
- [ ] 在 `_train_epoch` 和 `_validate` 中添加数据预处理
- [ ] 实现 targets list 到单一 tensor 的转换

### 步骤 7: 更新测试
- [ ] 添加单元测试

---

## 5. 数据流图

```mermaid
graph TD
    A[targets.json] --> B[TargetConfig list]
    B --> C[DataPreparator._extract_targets_and_write]
    C --> D[HDF5Writer.write_split]
    D --> E[/targets/{split}.h5]
    
    E --> F[/targets/0]
    E --> G[/targets/1]
    E --> H[/targets/2]
    
    F --> I[HDF5Backend.get_targets]
    I --> J[GenomicDataset.__getitem__]
    J --> K[Trainer]
    
    subgraph manifest.json
        L[num_targets: 3]
        M[lengths: 0: 896, 1: 1024, 2: 512]
        N[info: [...]]
    end
    
    style F fill:#e1f5fe
    style G fill:#e1f5fe
    style H fill:#e1f5fe
```

---

## 6. 关键接口

| 模块 | 接口 | 说明 |
|------|------|------|
| HDF5Writer.write_split | `targets: list[np.ndarray]` | targets 列表 |
| HDF5Writer.write_split | `target_configs: list[TargetConfig]` | 新增参数 |
| HDF5Backend.get_targets | `target_index: int \| None` | 新增参数 |
| HDF5Backend.get_targets | 返回 `torch.Tensor \| list[torch.Tensor]` | 支持返回列表 |
| GenomicDataset.__getitem__ | 返回 `tuple[Tensor, list[Tensor]]` | targets 变为列表 |
| GenomicDataset.get_batch | 返回 `tuple[Tensor, list[Tensor]]` | targets 变为列表 |
| DatasetMetadata | `target_lengths: dict[int, int]` | 新增字段 |

---

## 7. HDF5 结构

| 项目 | 说明 |
|------|------|
| 结构 | `/targets/{index}` |
| shape | `(num_seqs, target_length_i)` 2D |
| 元数据 | 仅 manifest.json |
| 变长支持 | 支持不同 target_length |
