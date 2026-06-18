# 新能源公交车队智能充电调度与绿色路径规划

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
make help       # 查看统一命令
make data       # 生成完整真实基础数据和派生实验数据
make data-small # 生成 3 条线路的小样本班次数据
make data-full  # 生成全量公交班次数据
make data-derived  # 生成车辆、充电站、电价、天气、能耗和路径数据
make data-validate # 校验处理后数据表兼容性
make simulate   # 生成模拟数据
make train      # 训练能耗预测模型
make jsq        # 运行 JSQ 基线调度
make ga         # 运行遗传算法优化调度
make evaluate   # 统计实验指标
make figures    # 生成图表
make run        # 运行完整实验流程
```

## 文档

- `新能源公交车队智能充电调度任务规划.md`：任务完成路径规划。
- `代码规范要求.md`：代码规范、并行化要求和运行规范。
- `docs/真实数据采集说明.md`：真实数据采集说明。

## 数据说明

当前仓库已经包含任务规划要求的数据表。公交班次来自香港 DATA.GOV.HK GTFS，天气来自香港天文台 API；车辆、公交可用充电站、能耗标签和路径候选是在真实班次基础上按项目假设构造，具体来源见 `docs/真实数据采集说明.md`。

能耗训练样本中的 `energy_kwh` 是模型训练使用的带噪声标签。生成流程先计算规则化的 `energy_kwh_clean`，再叠加基础高斯噪声、场景噪声和少量异常扰动；场景噪声会在高峰、拥堵、高载客、温度偏离舒适区、低速等条件下增大。这样可以避免模型只拟合过于干净的公式标签，使能耗预测任务更接近真实运营数据的不确定性。噪声生成按 `trip_id` 固定随机种子，可重复生成。

| 文件 | 内容 | 规模 |
|---|---|---:|
| `data/raw/hk_gtfs/gtfs.zip` | 原始 GTFS 压缩包 | 约 13 MB |
| `data/raw/hk_gtfs/extracted/` | 解压后的 GTFS 原始表 | 10 个文本表 |
| `data/raw/hko_current_weather.json` | 香港天文台当前天气原始 JSON | 1 个文件 |
| `data/processed/trips_hk_gtfs.csv` | 3 条线路的小样本班次数据 | 82 条班次 |
| `data/processed/trips_hk_gtfs_full.csv` | 全量公交班次数据 | 77,120 条班次 |
| `data/processed/vehicles.csv` | 车辆、电池容量、SOC 参数 | 20 辆车 |
| `data/processed/stations.csv` | 充电站候选点、桩数和功率 | 6 个站点 |
| `data/processed/prices.csv` | 分时电价 | 3 个时段 |
| `data/processed/weather_hourly.csv` | 小时级温度和湿度 | 24 小时 |
| `data/processed/energy_samples.csv` | 能耗预测训练样本，含干净能耗、带噪声能耗和噪声解释字段 | 77,120 条样本 |
| `data/processed/path_candidates.csv` | 绿色路径规划候选路径 | 231,360 条路径 |

重新生成完整数据：

```bash
make data
```
