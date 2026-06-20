# 不同电动化渗透率下的新能源公交充电调度优化研究

本仓库用于完成“案例7：新能源公交车队的智能充电调度与绿色路径规划”课程汇报项目。

项目目标：

1. 使用神经网络预测公交行程能耗。
2. 使用 JSQ 负载均衡策略作为传统基线方案。
3. 使用遗传算法优化充电调度。
4. 输出调度甘特图、SOC 曲线、成本对比和能耗对比分析。

## 环境

项目使用 Conda 环境：

```bash
make env-update
conda activate ebus-dispatch
```

## 常用命令

```bash
make help          # 查看统一命令
make scenario      # 查看当前场景对应的 JSQ/GA/评估输出路径
make data          # 生成班次、三阶段场景输入、派生数据并校验
make data-small    # 生成小样本班次数据
make data-full     # 生成全量班次数据
make data-derived  # 生成车辆、充电站、电价、天气、能耗和路径数据
make data-validate # 校验处理后数据表兼容性
make train         # 训练能耗预测模型
make jsq           # 运行 JSQ，默认 SCENARIO=full
make ga            # 运行遗传算法，默认 SCENARIO=full
make evaluate      # 统计 JSQ vs GA，默认 SCENARIO=full
make run-scenarios # 依次运行 current/planned/full 的 JSQ、GA 和评估
make scenario-summary # 汇总三阶段 JSQ/GA 指标表
make run           # 按顺序执行 data -> train -> jsq -> ga -> evaluate
```

实验统一使用 `SCENARIO` 参数选择电动化渗透率阶段：

```bash
make jsq SCENARIO=current
make ga SCENARIO=planned
make evaluate SCENARIO=full
```

GA 的适应度评估支持多进程并行，默认自动选择 worker 数；需要手动控制时可通过 `GA_ARGS` 覆盖：

```bash
make ga-full GA_ARGS="--parallel-workers 6"
make ga-current GA_ARGS="--parallel-workers 1"  # 关闭并行，便于调试
```

也可以使用场景别名：

```bash
make jsq-current      # 等价于 make jsq SCENARIO=current
make ga-planned       # 等价于 make ga SCENARIO=planned
make evaluate-full    # 等价于 make evaluate SCENARIO=full
```

班次数据规模可通过 Makefile 变量覆盖：

```bash
make data-small SMALL_ROUTE_COUNT=5 SMALL_TRIP_LIMIT=200
make data-full FULL_ROUTE_COUNT=99999 FULL_TRIP_LIMIT=999999
```

## 文档

- `新能源公交车队智能充电调度任务规划.md`：任务完成路径规划。
- `代码规范要求.md`：代码规范、并行化要求和运行规范。
- `docs/真实数据采集说明.md`：真实数据采集说明。

## 数据说明

当前仓库已经包含任务规划要求的数据表。公交班次来自香港 DATA.GOV.HK GTFS，天气来自香港天文台 API；车辆、公交可用充电站、能耗标签和路径候选是在真实班次基础上按项目假设构造，具体来源见 `docs/真实数据采集说明.md`。

调度实验改为三阶段电动化渗透率场景对比。三个场景按线路班次数从全量 GTFS 中选择高频线路簇，满足 `current ⊂ planned ⊂ full`，避免随机抽样导致线路结构失真。JSQ、遗传算法和评估都使用同一个 `SCENARIO` 参数：

| 场景 | 含义 | 车辆数 | 班次范围 | 作用 |
|---|---|---:|---|---|
| `current` | 当前试点/早期运营规模 | 150 | 按高频线路簇选取约 2.5% 全量班次 | 模拟现阶段小规模 e-bus 运营 |
| `planned` | 近期政策扩张规模 | 750 | 按高频线路簇选取约 12%-13% 全量班次 | 模拟资助采购新增 e-bus 后的中期规模 |
| `full` | 全面电动化覆盖 | 5,870 | 全量 77,120 班次 | 模拟未来全公交电动化场景 |

其中 `150 / 5870 ≈ 2.6%`，`750 / 5870 ≈ 12.8%`，`full` 为 100% 覆盖。`current` 可解释为当前注册电动公共巴士量级，`planned` 可解释为当前规模加上近期资助采购扩张后的规模。车辆、充电站和充电需求仍是围绕 GTFS 班次构造的研究场景，不是逐车真实运营清单；GTFS 本身只提供班次、线路和站点信息，不包含车辆排班和每辆车的实测能耗。

各场景输入文件如下：

| 场景 | 班次输入 | 车辆输入 | 充电站输入 | 常用命令 |
|---|---|---|---|---|
| `current` | `data/processed/trips_current.csv` | `data/processed/vehicles_current_150.csv` | `data/processed/stations_current.csv` | `make jsq-current`，`make ga-current`，`make evaluate-current` |
| `planned` | `data/processed/trips_planned.csv` | `data/processed/vehicles_planned_750.csv` | `data/processed/stations_planned.csv` | `make jsq-planned`，`make ga-planned`，`make evaluate-planned` |
| `full` | `data/processed/trips_full_coverage.csv` | `data/processed/vehicles_full_5870.csv` | `data/processed/stations_full_80hubs.csv` | `make jsq-full`，`make ga-full`，`make evaluate-full` |

各场景输出文件保持同一套命名规则：

| 场景 | JSQ 输出 | GA 输出 | 评估输出 |
|---|---|---|---|
| `current` | `outputs/schedules/jsq_schedule_current.csv`，`outputs/metrics/jsq_metrics_current.json` | `outputs/schedules/ga_schedule_current.csv`，`outputs/metrics/ga_metrics_current.json` | `outputs/metrics/evaluation_summary_current.json` |
| `planned` | `outputs/schedules/jsq_schedule_planned.csv`，`outputs/metrics/jsq_metrics_planned.json` | `outputs/schedules/ga_schedule_planned.csv`，`outputs/metrics/ga_metrics_planned.json` | `outputs/metrics/evaluation_summary_planned.json` |
| `full` | `outputs/schedules/jsq_schedule_full.csv`，`outputs/metrics/jsq_metrics_full.json` | `outputs/schedules/ga_schedule_full.csv`，`outputs/metrics/ga_metrics_full.json` | `outputs/metrics/evaluation_summary_full.json` |

三阶段总表由 `make scenario-summary` 生成：

| 文件 | 内容 |
|---|---|
| `outputs/metrics/penetration_scenario_summary.csv` | current/planned/full × JSQ/GA 的完成率、成本、等待时间、充电次数、充电电量和间接碳排放等对比表 |
| `outputs/metrics/penetration_scenario_summary.json` | 同一汇总表的 JSON 版本 |

碳排放口径采用充电用电的间接排放：

```text
total_charging_co2_kg = total_charged_energy_kwh × 0.55 kgCO2/kWh
```

该系数与路径候选数据中的 `carbon_kgco2` 计算保持一致。由于电动巴士运行阶段没有尾气排放，这里比较的是不同调度策略造成的充电电量差异和对应电力侧排放差异。

能耗训练样本中的 `energy_kwh` 是模型训练使用的带噪声标签。生成流程先计算规则化的 `energy_kwh_clean`，再叠加基础高斯噪声、场景噪声和少量异常扰动；场景噪声会在高峰、拥堵、高载客、温度偏离舒适区、低速等条件下增大。这样可以避免模型只拟合过于干净的公式标签，使能耗预测任务更接近真实运营数据的不确定性。噪声生成按 `trip_id` 固定随机种子，可重复生成。

| 文件 | 内容 | 规模 |
|---|---|---:|
| `data/raw/hk_gtfs/gtfs.zip` | 原始 GTFS 压缩包 | 约 13 MB |
| `data/raw/hk_gtfs/extracted/` | 解压后的 GTFS 原始表 | 10 个文本表 |
| `data/raw/hko_current_weather.json` | 香港天文台当前天气原始 JSON | 1 个文件 |
| `data/processed/trips_hk_gtfs.csv` | 3 条线路的小样本班次数据 | 82 条班次 |
| `data/processed/trips_hk_gtfs_full.csv` | 全量公交班次数据 | 77,120 条班次 |
| `data/processed/trips_current.csv` | current 场景高频线路簇班次 | 约 2.5% 全量班次 |
| `data/processed/trips_planned.csv` | planned 场景高频线路簇班次 | 约 12%-13% 全量班次 |
| `data/processed/trips_full_coverage.csv` | full 场景全覆盖班次 | 77,120 条班次 |
| `data/processed/vehicles.csv` | 车辆、电池容量、SOC 参数 | 20 辆车 |
| `data/processed/vehicles_current_150.csv` | current 场景合成车辆、电池容量、SOC 参数 | 150 辆车 |
| `data/processed/vehicles_planned_750.csv` | planned 场景合成车辆、电池容量、SOC 参数 | 750 辆车 |
| `data/processed/vehicles_full_5870.csv` | full 场景合成车辆、电池容量、SOC 参数 | 5,870 辆车 |
| `data/processed/stations.csv` | 充电站候选点、桩数和功率 | 6 个站点 |
| `data/processed/stations_current.csv` | current 场景充电站候选点、桩数和功率 | 6 个站点 |
| `data/processed/stations_planned.csv` | planned 场景充电站候选点、桩数和功率 | 20 个站点 |
| `data/processed/stations_full_80hubs.csv` | full 场景合成充电枢纽、桩数和功率 | 80 个站点 |
| `data/processed/prices.csv` | 分时电价 | 3 个时段 |
| `data/processed/weather_hourly.csv` | 小时级温度和湿度 | 24 小时 |
| `data/processed/energy_samples.csv` | 能耗预测训练样本，含干净能耗、带噪声能耗和噪声解释字段 | 77,120 条样本 |
| `data/processed/path_candidates.csv` | 绿色路径规划候选路径 | 231,360 条路径 |

重新生成完整数据：

```bash
make data
```

## 图表与汇报解读

所有图表由以下命令生成：

```bash
make scenario-summary
make figures
```

图表统一输出到 `outputs/figures/`。建议汇报顺序是：

```text
能耗预测对比
    ↓
绿色路径规划
    ↓
三阶段调度结果
    ↓
调度过程可视化
```

### 1. 能耗预测对比图

任务书要求比较“固定单位里程耗电模型”和“神经网络能耗预测模型”。本项目不需要重新训练 baseline，而是直接基于 `data/processed/energy_predictions.csv` 计算固定里程模型：

```text
fixed_energy_kwh = distance_km × 全样本平均单位里程耗电
```

其中：

```text
全样本平均单位里程耗电 = sum(energy_kwh) / sum(distance_km)
```

指标输出：

| 文件 | 内容 |
|---|---|
| `outputs/metrics/energy_prediction_comparison.json` | 固定里程模型和神经网络模型的 MAE、RMSE、R² 对比 |

当前结果：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 固定单位里程耗电模型 | 5.390 | 8.413 | 0.811 |
| 神经网络能耗预测模型 | 2.854 | 4.763 | 0.939 |

结论可以表述为：

```text
相比固定单位里程耗电模型，神经网络模型将 MAE 降低约 47.0%，将 RMSE 降低约 43.4%，R² 从 0.811 提高到 0.939。说明引入坡度、拥堵、速度、载客率、温度和高峰时段等特征后，模型能更准确预测单趟公交能耗。
```

相关图表：

| 图表 | 读法 | 汇报用途 |
|---|---|---|
| `energy_model_metric_comparison.png` | 横向比较两个模型的 MAE、RMSE、R²。MAE/RMSE 越低越好，R² 越高越好。 | 证明神经网络模型优于简单固定里程模型。 |
| `energy_actual_vs_predicted.png` | 横轴是真实能耗，纵轴是神经网络预测能耗；点越接近对角线，预测越准确。 | 展示模型预测值整体贴近真实标签。 |
| `energy_prediction_error_distribution.png` | 横轴是预测误差，0 附近越集中越好；比较固定里程模型和神经网络误差分布。 | 展示神经网络误差更集中，极端偏差更少。 |

### 2. 绿色路径规划图

绿色路径规划使用 `data/processed/path_candidates.csv`。每个 `trip_id` 有 3 条候选路径，包含距离、能耗和碳排放：

```text
shorter_with_more_slope
balanced_low_congestion
longer_but_flatter
```

本项目做两类对比：

1. 逐班次选择“最短距离路径”。
2. 逐班次选择“最低能耗路径”。
3. 按候选路径类型聚合，比较不同路径类型的总距离、总能耗和总碳排。

指标输出：

| 文件 | 内容 |
|---|---|
| `outputs/metrics/green_path_summary.json` | 最短距离策略、最低能耗策略和路径类型聚合结果 |
| `outputs/metrics/green_path_summary.csv` | 最短距离策略 vs 最低能耗策略汇总表 |
| `outputs/metrics/green_path_type_summary.csv` | 三类候选路径的距离、能耗、碳排放汇总表 |

当前结果中，`99.98%` 的班次里最短距离路径同时也是最低能耗路径，所以“最短距离 vs 最低能耗”的差异很小。这不是错误，而是当前候选路径生成规则下，距离仍然是能耗的主导因素。

更适合汇报的结论是：

```text
在当前候选路径集合中，最短路径大多数情况下也是最低能耗路径，说明公交行程能耗主要受行驶距离驱动。路径类型对比进一步显示，较长候选路径会带来更高总能耗和更高间接碳排放，因此绿色路径规划可以作为调度前的路径筛选环节，避免不必要的绕行能耗。
```

相关图表：

| 图表 | 读法 | 汇报用途 |
|---|---|---|
| `green_path_strategy_comparison.png` | 比较“最短距离路径”和“最低能耗路径”的总距离、总能耗、总碳排。 | 说明严格按最低能耗选路与最短路径结果接近。 |
| `green_path_type_comparison.png` | 比较三类候选路径的总距离、总能耗、总碳排。 | 展示路径类型不同会影响能耗和碳排，支撑“绿色路径规划”主线。 |

### 3. 三阶段调度结果图

三阶段场景为：

```text
current：当前试点规模，150 辆车，1713 个班次
planned：近期扩张规模，750 辆车，9717 个班次
full：全面覆盖规模，5870 辆车，77120 个班次
```

三阶段汇总表：

| 文件 | 内容 |
|---|---|
| `outputs/metrics/penetration_scenario_summary.csv` | current/planned/full × JSQ/GA 的核心指标表 |
| `outputs/metrics/penetration_scenario_summary.json` | 同一结果的 JSON 版本 |

当前主要结果：

| 场景 | 结论 |
|---|---|
| current | 规模较小，JSQ 和 GA 都能完成全部班次，且没有触发充电事件，说明当前试点规模下充电调度压力不明显。 |
| planned | JSQ 和 GA 都完成全部班次；GA 将充电成本从约 119,652 降至 59,937，成本降低约 49.9%。 |
| full | JSQ 和 GA 都完成全部班次；GA 将充电成本从约 1,582,114 降至 1,016,219，成本降低约 35.8%；平均等待时间从 11.50 分钟降至 2.93 分钟。 |

相关图表：

| 图表 | 读法 | 汇报用途 |
|---|---|---|
| `scenario_cost_savings.png` | 展示 planned 和 full 场景下 GA 相比 JSQ 节省的充电成本。 | 最直接展示 GA 的经济收益。 |
| `scenario_full_wait_reduction.png` | 展示 full 场景下 GA 对平均等待时间和最大等待时间的改善。 | 说明 GA 不只省钱，也改善充电排队效率。 |
| `scenario_normalized_efficiency.png` | 按每 1000 个班次归一化比较成本和充电电量。 | 避免 current/planned/full 因规模差异过大而无法直接比较。 |
| `scenario_report_dashboard.png` | 2×2 综合看板，包含规模、成本下降、等待下降、碳排下降。 | 最适合作为汇报总览页。 |
| `scenario_cost_comparison.png` | 三阶段 JSQ 和 GA 总成本柱状对比。 | 直接比较不同阶段总成本。 |
| `scenario_wait_time_comparison.png` | 三阶段 JSQ 和 GA 平均等待时间对比。 | 展示充电排队压力随规模上升而增加，GA 在 full 场景更有价值。 |
| `scenario_charging_events.png` | 三阶段 JSQ 和 GA 充电事件数量对比。 | 展示电动化规模扩大后，充电调度任务数量明显增加。 |
| `scenario_completion_rate.png` | 三阶段 JSQ 和 GA 完成率对比。 | 证明在当前参数下，两种策略都能满足班次完成率约束。 |

建议汇报表述：

```text
随着电动公交规模从 current 增长到 planned 和 full，充电调度问题逐渐从低压力场景变为高压力场景。GA 在 planned 和 full 场景下都能在保证班次完成率的同时显著降低充电成本；在 full 场景下还明显降低平均等待时间，说明智能优化策略在大规模电动化后更有应用价值。
```

### 4. 调度过程可视化图

调度过程图用于说明“调度不是只算一个总成本，而是生成了可执行的充电计划”。

相关图表：

| 图表 | 读法 | 汇报用途 |
|---|---|---|
| `jsq_charging_gantt_full.png` | JSQ full 场景的充电甘特图。每条横条是一条充电任务；横轴是时间；横条长度是充电持续时间；橙色是快充，蓝色是慢充；条内文字为车辆和充电桩。 | 展示 JSQ 的具体充电任务安排。 |
| `ga_charging_gantt_full.png` | GA full 场景的充电甘特图，读法同上。 | 展示 GA 优化后的具体充电任务安排。 |
| `charging_load_curve_full.png` | 横轴是全天时间，纵轴是同时处于充电状态的总功率，单位 MW；蓝线是 JSQ，橙线是 GA。 | 比甘特图更适合汇报，用于展示全天充电负荷分布差异。 |
| `jsq_soc_curves_full.png` | 选取 full 场景中任务较多的车辆，绘制 JSQ 下 SOC 随时间变化曲线；虚线为 20% 安全 SOC。 | 展示车辆电量始终受安全阈值约束。 |
| `ga_soc_curves_full.png` | 选取 full 场景中任务较多的车辆，绘制 GA 下 SOC 随时间变化曲线；虚线为 20% 安全 SOC。 | 展示 GA 更激进地利用电池容量，但仍不低于安全阈值。 |
| `ga_convergence_full.png` | 横轴为 GA 迭代代数，纵轴为最优充电成本和适应度。曲线下降代表搜索逐步找到更优策略。 | 证明遗传算法确实发生了优化收敛过程。 |

甘特图读法说明：

```text
一条横条 = 一次充电事件
横条左端 = 开始充电时间
横条右端 = 结束充电时间
横条长度 = 充电持续时间
橙色 = 快充
蓝色 = 慢充
条内文字 = 车辆编号 / 充电桩编号
```

由于 full 场景充电事件很多，甘特图只截取前若干个充电事件作为样例。汇报时不建议把甘特图作为主要结论图，它更适合说明“调度结果可执行”。真正更适合展示整体调度差异的是：

```text
charging_load_curve_full.png
```

可以这样讲：

```text
甘特图展示了单次充电任务的起止时间和充电桩分配。由于全量场景充电事件较多，本文进一步使用全天充电负荷曲线展示 JSQ 与 GA 的整体调度差异。负荷曲线反映不同策略下充电需求在一天内的分布变化，比单个甘特图更适合表达系统层面的调度效果。
```

### 5. 图表选择建议

如果汇报时间有限，建议只放以下 6 张图：

| 顺序 | 图表 | 目的 |
|---:|---|---|
| 1 | `energy_model_metric_comparison.png` | 证明神经网络能耗预测优于固定里程 baseline。 |
| 2 | `energy_actual_vs_predicted.png` | 直观展示预测值贴近真实能耗。 |
| 3 | `green_path_type_comparison.png` | 补齐绿色路径规划主线。 |
| 4 | `scenario_report_dashboard.png` | 总览三阶段调度优化结果。 |
| 5 | `charging_load_curve_full.png` | 展示 JSQ 和 GA 全天充电负荷差异。 |
| 6 | `ga_convergence_full.png` | 展示遗传算法优化过程。 |

如果需要更详细的附录，再加入：

```text
energy_prediction_error_distribution.png
green_path_strategy_comparison.png
jsq_charging_gantt_full.png
ga_charging_gantt_full.png
jsq_soc_curves_full.png
ga_soc_curves_full.png
```
