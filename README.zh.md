# TrajStat（Python 版本）

TrajStat 是一个用于处理大气回轨轨迹的工具包。本仓库在保持原 Java 插件整体架构的前提下，以 Python 3.9+ 语言特性重写了核心功能，提供等价的轨迹转换、聚类统计与潜在源贡献分析能力。

## 功能概览

- 使用 `TrajConfig` 对象组织 HYSPLIT 控制文件，批量触发外部回轨计算。
- 将 HYSPLIT 输出的端点文本转换为 TrajStat (TGS) CSV，并支持批量转换、合并与 GeoJSON 轨迹导出。
- 在 `VectorLayer` 之上实现数据增强、聚类统计、PSCF 与 CWT 计算等工具方法，可通过 CLI 或 `TrajStatPlugin` API 直接调用。
- 使用 Typer 构建 CLI，Fiona/Shapely 负责矢量数据读写与几何运算，NumPy/Pydantic 等库提供数值与配置能力。

## 计算公式解析

### 1. 聚类统计

聚类统计以轨迹集合 \(\{T_k\}_{k=1}^K\) 与用户给定的聚类结果 \(c_k\) 为输入。`trajstat.trajectory.util.calculate_cluster_statistics` 会为每个聚类 \(g\) 计算：

- **平均轨迹** \(\bar{T}_g\)：对同一聚类内所有轨迹端点坐标进行逐时平均：
  \[
  \bar{T}_g(t) = \frac{1}{|G|} \sum_{k\in G} T_k(t)
  \]
- **总空间方差** (TSV)：衡量聚类内部离散度，具体实现中以欧氏距离或角度距离累积每条轨迹相对平均轨迹的平方差。

平均轨迹与 TSV 的求解逻辑保持与原 Java 版本一致，但通过 Python 的数据类与列表推导实现，提升了代码可读性。

### 2. PSCF（Potential Source Contribution Function）

PSCF 衡量特定栅格单元对污染超标事件的潜在贡献。`calculate_endpoint_counts` 先统计：

- \(N_{ij}\)：轨迹端点落入第 \(i,j\) 个栅格的次数；
- \(M_{ij}\)：满足阈值条件（如观测值大于设定临界值）的轨迹中，端点落入该栅格的次数。

随后 `calculate_pscf_field` 按公式
\[
\text{PSCF}_{ij} = \frac{M_{ij}}{N_{ij}}, \quad N_{ij} > 0
\]
计算每个单元的 PSCF，若 \(N_{ij}=0\) 则结果为 0。实现中使用 `_point_in_polygon` 判断端点是否落入栅格，并通过集合跟踪避免对同一条轨迹重复计数。

#### 权重修正

当 \(N_{ij}\) 较小（统计样本不足）时，可使用 `weight_by_counts` 按阈值 \(T_1>T_2>\dots>T_n\) 与权重 \(W_1, W_2, \dots, W_n\) 调整：
\[
\text{PSCF}_{ij}' = \text{PSCF}_{ij} \times W_m, \quad T_{m+1} < N_{ij} \le T_m
\]
若 \(N_{ij} \le T_n\) 则直接乘以 \(W_n\)。权重表可以来自文献经验或用户配置。

### 3. CWT（Concentration Weighted Trajectory）

CWT 用以评估污染物平均浓度与轨迹空间位置的关系。`calculate_cwt_field` 将轨迹端点携带的观测值 \(C_k\) 投影到栅格：
\[
\text{CWT}_{ij} = \frac{\sum_{k \in S_{ij}} C_k}{|S_{ij}|}
\]
其中 \(S_{ij}\) 为落入栅格 \((i,j)\) 的端点集合。实现中，若某栅格没有端点落入则返回 0。与传统定义 \(\sum_k C_k \tau_{ijk}/\sum_k \tau_{ijk}\) 一致，当端点驻留时间 \(\tau_{ijk}\) 统一时即退化为端点平均值。

## 实现过程解析

1. **模块划分**：保留原 Java 项目的分层结构，将轨迹配置、控制和工具函数分别放在 `trajstat/trajectory` 子包中，CLI 与插件外观层分别位于 `trajstat/cli.py` 与 `trajstat/main.py`。
2. **矢量数据封装**：`VectorLayer` 对 Fiona 的 `Collection` 进行轻量封装，提供字段检查、属性表写回、矢量几何迭代等接口，方便与旧版 API 对接。
3. **计算核心重写**：PSCF/CWT、聚类等算法以纯 Python 实现，结合类型提示和单元测试 (`tests/test_traj_util.py`) 确保行为与 Java 版本一致。
4. **CLI 与插件**：借助 Typer 生成命令行，同时暴露 `TrajStatPlugin` 以便脚本或服务端调用，实现图形界面缺失情况下的自动化处理。
5. **依赖替换**：使用 Fiona/Shapely 取代 MeteoInfo 的 GIS 功能，Typer 取代 Swing 命令入口，SQLAlchemy 等可按需补充（当前模块未涉及数据库操作）。

## 使用示例

### 命令行

```bash
trajstat cluster-stats month.tgs 1 5 --point-count 24
trajstat pscf-counts grid.geojson month.tgs --value-field SO2 --missing-value -9999
trajstat pscf grid.geojson --nij-field Nij --mij-field Mij --output-field PSCF
trajstat cwt grid.geojson month.tgs --value-field SO2 --missing-value -9999
trajstat weight grid.geojson --base-field PSCF --count-field Nij \
  --target-field WPSCF --thresholds 3 5 10 --ratios 0.2 0.4 0.7
```

### Python API

```python
from pathlib import Path

from trajstat.main import TrajStatPlugin
from trajstat.vector import VectorLayer

plugin = TrajStatPlugin()
traj_layer = VectorLayer.from_geojson(Path("month.geojson"))
polygon = VectorLayer.from_geojson(Path("grid.geojson"))

nij, mij = plugin.fill_endpoint_counts(
    [traj_layer],
    polygon,
    value_field="SO2",
    missing_value=-9999,
    count_field="Nij",
    trajectory_count_field="Mij",
)
pscf = plugin.calculate_pscf(polygon, nij_field="Nij", mij_field="Mij")
weighted = plugin.weight_by_counts(
    polygon,
    base_field="PSCF",
    count_field="Nij",
    target_field="WPSCF",
    thresholds=[10, 5, 3],
    ratios=[0.7, 0.4, 0.2],
)
cwt, counts = plugin.calculate_cwt(
    [traj_layer],
    polygon,
    value_field="SO2",
    missing_value=-9999,
)
```

## 测试与验证

项目提供 pytest 用例覆盖轨迹转换、PSCF/CWT 统计等核心流程：

```bash
pytest
```

## 文档导航

- [English README](README.md)
- [API 参考示例](trajstat/main.py)
- [测试样例](tests/test_traj_util.py)

欢迎提交 Issue 或 PR 以完善功能及文档。
