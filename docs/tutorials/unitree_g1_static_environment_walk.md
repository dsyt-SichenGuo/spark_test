# Unitree G1 静态环境行走

在仓库根目录使用 `spark_isaac` 环境运行统一高层入口：

```bash
conda activate spark_isaac
python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py --environment marble
```

`--environment marble` 加载 `reconstructions/marble-1.1-static`；
`--environment gpt-v4` 加载 `reconstructions/gpt-v4-static`。
资产选择、场景 Z 偏移及默认参数集中在高层脚本的 `ENVIRONMENT_ASSETS` 中，
两个场景使用同一个 tensor 运行时。静态资产不包含动态人体、人体网格回放、
回放相机或时间采样动画；运行时还会校验静态场景和碰撞几何。
USD 环境任务关闭 SPARK 自动生成的装饰地面网格，并移除默认平面，
使用资产自带的地面与碰撞几何，避免 Z=0 的装饰网格悬在重建地面上方。

原 `run_unitree_g1_marble_high_level.py` 入口保留，支持相同的参数。

## 初始化模式

所有位置参数均为**环境资产的局部 XY 坐标，单位米**，不包含多环境克隆的平移。
初始朝向用度表示：`0` 朝 +X，`90` 朝 +Y。

| 独立模式参数 | 随机模式（默认） | 手动模式 |
| --- | --- | --- |
| `--robot-position-mode` | `random` | `manual --robot-position X Y` |
| `--robot-yaw-mode` | `random`，范围 −180° 到 180° | `manual --robot-yaw-degrees DEG` |
| `--goal-position-mode` | `random` | `manual --goal-position X Y` |

Marble 和 GPT-v4 均默认三个配置全部随机。三个模式可任意组合；
也可以省略模式参数，直接提供对应坐标或角度，此时自动选择手动模式。
显式选择 `random` 时不能同时提供对应手动值；选择 `manual` 必须提供值。
每次成功、跌倒、卡住或超时重置时，随机项重新采样，手动项保持不变。
机器人与目标的初始距离至少为 `--min-goal-distance`（默认 3 米），
两个位置均手动时也会检查该距离。
“初始位姿”在这里指初始 yaw；roll/pitch 保持直立，Z 仍按静态地面校准。

机器人实时朝向**当前位置指向目标位置的方向**，
即 `atan2(goal_y - robot_y, goal_x - robot_x)`。
成功需要同时满足位置误差不超过 `--goal-tolerance`（默认 0.45 米）和
朝向误差不超过 `--goal-yaw-tolerance-degrees`（默认 15°）。
当机器人与目标 XY 距离不超过 0.1 毫米，方向无法由当前位置定义，
此时沿用本回合起点到目标的方向。进入目标位置范围后允许原地转向，
不会因为 XY 位移小而被判为卡住。

全部随机（两个环境均适用）：

```bash
python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py \
  --environment marble --seed 42
```

手动初始位置，随机朝向和目标：

```bash
python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py \
  --environment marble --robot-position -0.2 -1.0
```

随机初始位置，手动朝向和目标：

```bash
python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py \
  --environment gpt-v4 --robot-yaw-degrees 90 --goal-position -4 -4
```

三个配置均手动（使用显式模式）：

```bash
python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py \
  --environment marble \
  --robot-position-mode manual --robot-position -0.2 -1.0 \
  --robot-yaw-mode manual --robot-yaw-degrees 217.24 \
  --goal-position-mode manual --goal-position 2 2
```

手动坐标还需满足资产地面支撑检查；示例目标需要按实际场景选择。
旧 `--init-mode random|fixed-start|fixed-goal` 命令继续兼容，
分别映射为全部随机、手动起点及朝向、手动目标。

随机采样默认使用场景 XY 边界向内收缩 `--wall-margin`（默认 5 米）的矩形。
可用 `--sampling-bounds XMIN XMAX YMIN YMAX` 指定场景内的采样矩形，
该参数优先于 `--wall-margin`。固定坐标可以在采样矩形之外，但必须位于场景边界内。
Marble 额外检查地面的支撑、坡度和局部高度变化，过滤没有支撑的起点和目标，
并自动将机器人基座放到对应地面上方 0.793 米。
`--spawn-height Z` 可以显式设置对齐后的基座高度。
跌倒高度阈值 `--fall-height` 相对于出生点地面计算。
当前任务使用原来的目标跟踪控制器，遇到墙体或货架阻挡时仍通过卡住判定重置回合。

## 视角刷新

每次初始化和重置后，视角在机器人初始位置与目标位置的 XY 连线上，
从机器人初始位置向目标的反方向退后，朝目标方向观察。
视角依据起点与目标的连线设置，不受随机机器人初始朝向影响。
Marble 会检查相机与场景网格的相交情况：遇到屋顶时降低相机，遇到墙体时
缩短后退距离，并限制相机位于场景 XY 边界内。相机拉近后，瞄准点也会靠近
机器人躯干，避免机器人落到画面之外。因此下表中的相机距离和高度是期望值，
实际视角会根据室内空间调整。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--camera-distance` | 3.0 米 | 相机位于初始位置后方的水平距离 |
| `--camera-height` | 2.0 米 | 相机高于初始基座的距离 |
| `--camera-lookahead` | 1.5 米 | 相机瞄准点沿起点到目标方向的前移距离 |

多环境共用一个活动视口；同时有多行重置时，刷新到其中第一行的起点。
无界面模式不设置视口。

Marble 室内另外配置了均匀补光和相机附近的补光，避免封闭屋顶挡住穹顶光后
场景内部过暗。`--interior-light-intensity`（默认 1600）控制室内补光亮度，
`--camera-light-intensity`（默认 1000）控制相机附近的补光亮度，两者可设为 0。
`--interior-ambient-intensity`（默认 0.8）控制 RTX 室内环境补光，
需要更亮时可以调大，例如 `--interior-ambient-intensity 1.2`。

## 批量测试与命令预览

```bash
python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py \
  --environment gpt-v4 --policy sportsafe \
  --num-envs 4 --headless --steps 300 --seed 42

python example/unitree_g1/run_unitree_g1_usd_environment_walk_benchmark.py \
  --environment marble --dry-run
```

`--policy wbtsafe` / `--policy sportsafe` 沿用原脚本的控制器名称和行为；
也可用 benchmark 风格的 `--policy-config UnitreeG1WBTSafePolicy` 或
`--policy-config UnitreeG1SportSafePolicy`。这些名称不会额外添加安全过滤器。
