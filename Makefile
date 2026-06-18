SHELL := /bin/bash
.DEFAULT_GOAL := help

CONDA_ENV ?= ebus-dispatch
CONDA ?= conda
RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON := $(RUN) python

.PHONY: help env env-update env-remove activate setup dirs data data-small data-full data-derived data-validate simulate train jsq ga evaluate figures run test format lint clean

help:
	@echo "新能源公交车队智能充电调度项目统一命令"
	@echo ""
	@echo "环境管理:"
	@echo "  make env          创建 Conda 环境: $(CONDA_ENV)"
	@echo "  make env-update   根据 environment.yml 更新环境"
	@echo "  make env-remove   删除 Conda 环境"
	@echo "  make activate     显示环境激活命令"
	@echo "  make setup        创建输出目录"
	@echo ""
	@echo "数据与实验:"
	@echo "  make data         生成完整真实基础数据和派生实验数据"
	@echo "  make data-small   生成 3 条线路的小样本班次数据"
	@echo "  make data-full    生成全量公交班次数据"
	@echo "  make data-derived 生成车辆、充电站、电价、天气、能耗和路径数据"
	@echo "  make data-validate 校验处理后数据表兼容性"
	@echo "  make simulate     生成模拟数据"
	@echo "  make train        训练能耗预测模型"
	@echo "  make jsq          运行 JSQ 基线调度"
	@echo "  make ga           运行遗传算法优化调度"
	@echo "  make evaluate     统计实验指标"
	@echo "  make figures      生成图表"
	@echo "  make run          运行完整实验流程"
	@echo ""
	@echo "质量检查:"
	@echo "  make test         运行测试"
	@echo "  make format       格式化代码"
	@echo "  make lint         静态检查"
	@echo "  make clean        清理缓存文件"

env:
	$(CONDA) env create -f environment.yml

env-update:
	$(CONDA) env update -n $(CONDA_ENV) -f environment.yml --prune

env-remove:
	$(CONDA) env remove -n $(CONDA_ENV)

activate:
	@echo "conda activate $(CONDA_ENV)"

setup: dirs

dirs:
	@mkdir -p data/raw data/processed data/simulated
	@mkdir -p outputs/figures outputs/schedules outputs/metrics outputs/models
	@mkdir -p docs tests src

data: data-full data-small data-derived data-validate

data-small:
	@if [ -f scripts/collect_hk_gtfs.py ]; then \
		$(PYTHON) scripts/collect_hk_gtfs.py; \
	else \
		echo "未找到 scripts/collect_hk_gtfs.py，跳过真实数据采集。"; \
	fi

data-full:
	@if [ -f scripts/collect_hk_gtfs.py ]; then \
		$(PYTHON) scripts/collect_hk_gtfs.py --max-routes 99999 --max-trips 999999 --output data/processed/trips_hk_gtfs_full.csv; \
	else \
		echo "未找到 scripts/collect_hk_gtfs.py，跳过全量 GTFS 数据处理。"; \
	fi

data-derived:
	@if [ -f scripts/build_project_datasets.py ]; then \
		$(PYTHON) scripts/build_project_datasets.py; \
	else \
		echo "未找到 scripts/build_project_datasets.py，跳过派生实验数据生成。"; \
	fi

data-validate:
	@if [ -f scripts/validate_datasets.py ]; then \
		$(PYTHON) scripts/validate_datasets.py; \
	else \
		echo "未找到 scripts/validate_datasets.py，跳过数据兼容性校验。"; \
	fi

simulate:
	@if [ -f src/simulate_data.py ]; then \
		$(PYTHON) -m src.simulate_data; \
	else \
		echo "未找到 src/simulate_data.py，请先实现模拟数据模块。"; \
	fi

train:
	@if [ -f src/energy_model.py ]; then \
		$(PYTHON) -m src.energy_model; \
	else \
		echo "未找到 src/energy_model.py，请先实现能耗预测模块。"; \
	fi

jsq:
	@if [ -f src/baseline_jsq.py ]; then \
		$(PYTHON) -m src.baseline_jsq; \
	else \
		echo "未找到 src/baseline_jsq.py，请先实现 JSQ 基线模块。"; \
	fi

ga:
	@if [ -f src/ga_optimizer.py ]; then \
		$(PYTHON) -m src.ga_optimizer; \
	else \
		echo "未找到 src/ga_optimizer.py，请先实现遗传算法模块。"; \
	fi

evaluate:
	@if [ -f src/evaluation.py ]; then \
		$(PYTHON) -m src.evaluation; \
	else \
		echo "未找到 src/evaluation.py，请先实现评估模块。"; \
	fi

figures:
	@if [ -f src/visualization.py ]; then \
		$(PYTHON) -m src.visualization; \
	else \
		echo "未找到 src/visualization.py，请先实现可视化模块。"; \
	fi

run: setup
	@if [ -f src/main.py ]; then \
		$(PYTHON) -m src.main; \
	else \
		echo "未找到 src/main.py，请先实现主程序入口。"; \
	fi

test:
	$(RUN) pytest -q

format:
	$(RUN) black src tests scripts
	$(RUN) isort src tests scripts

lint:
	$(RUN) ruff check src tests scripts

clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
