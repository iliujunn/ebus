SHELL := /bin/bash
.DEFAULT_GOAL := help

CONDA_ENV ?= ebus-dispatch
CONDA ?= conda
RUN := $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PYTHON := $(RUN) python

SCENARIO ?= full
SCENARIOS := current planned full small_demo hk_scale default
PRIMARY_SCENARIOS := current planned full

SMALL_ROUTE_COUNT ?= 3
SMALL_TRIP_LIMIT ?= 1000
FULL_ROUTE_COUNT ?= 99999
FULL_TRIP_LIMIT ?= 999999

TRIPS_SMALL := data/processed/trips_hk_gtfs.csv
TRIPS_FULL := data/processed/trips_hk_gtfs_full.csv

SCENARIO_SUFFIX = $(if $(filter current,$(SCENARIO)),_current,$(if $(filter planned,$(SCENARIO)),_planned,$(if $(filter full,$(SCENARIO)),_full,$(if $(filter small_demo,$(SCENARIO)),_small_demo,$(if $(filter hk_scale,$(SCENARIO)),_hk_scale,)))))
EVALUATION_SUFFIX = $(if $(filter default,$(SCENARIO)),_default,$(SCENARIO_SUFFIX))
JSQ_METRICS = outputs/metrics/jsq_metrics$(SCENARIO_SUFFIX).json
GA_METRICS = outputs/metrics/ga_metrics$(SCENARIO_SUFFIX).json
JSQ_SCHEDULE = outputs/schedules/jsq_schedule$(SCENARIO_SUFFIX).csv
GA_SCHEDULE = outputs/schedules/ga_schedule$(SCENARIO_SUFFIX).csv
EVALUATION_OUTPUT ?= outputs/metrics/evaluation_summary$(EVALUATION_SUFFIX).json
SCENARIO_SUMMARY_CSV ?= outputs/metrics/penetration_scenario_summary.csv
SCENARIO_SUMMARY_JSON ?= outputs/metrics/penetration_scenario_summary.json

GA_ARGS ?=

.PHONY: help check-scenario scenario env env-update env-remove activate setup dirs \
	data data-small data-full data-derived data-validate simulate train \
	jsq jsq-current jsq-planned jsq-full jsq-small jsq-hk jsq-default \
	ga ga-current ga-planned ga-full ga-small ga-hk ga-default \
	evaluate evaluate-current evaluate-planned evaluate-full evaluate-small evaluate-hk evaluate-default \
	scenario-summary run-scenarios figures run experiment \
	test format lint clean

help:
	@echo "新能源公交车队智能充电调度项目统一命令"
	@echo ""
	@echo "场景参数:"
	@echo "  SCENARIO=current   当前试点规模：约 150 辆车 + 约 2.5% 班次"
	@echo "  SCENARIO=planned   近期扩张规模：约 750 辆车 + 约 12%-13% 班次"
	@echo "  SCENARIO=full      全面覆盖规模：5870 辆车 + 全量班次（默认）"
	@echo "  legacy: small_demo hk_scale default"
	@echo ""
	@echo "数据规模参数:"
	@echo "  SMALL_ROUTE_COUNT=$(SMALL_ROUTE_COUNT) SMALL_TRIP_LIMIT=$(SMALL_TRIP_LIMIT)"
	@echo "  FULL_ROUTE_COUNT=$(FULL_ROUTE_COUNT) FULL_TRIP_LIMIT=$(FULL_TRIP_LIMIT)"
	@echo ""
	@echo "环境管理:"
	@echo "  make env             创建 Conda 环境: $(CONDA_ENV)"
	@echo "  make env-update      根据 environment.yml 更新环境"
	@echo "  make env-remove      删除 Conda 环境"
	@echo "  make activate        显示环境激活命令"
	@echo "  make setup           创建输出目录"
	@echo ""
	@echo "数据与模型:"
	@echo "  make data            生成班次、三阶段场景输入、派生数据并校验"
	@echo "  make data-small      生成小样本班次数据"
	@echo "  make data-full       生成全量班次数据"
	@echo "  make data-derived    生成车辆、充电站、电价、天气、能耗和路径数据"
	@echo "  make data-validate   校验处理后数据表兼容性"
	@echo "  make train           训练能耗预测模型"
	@echo ""
	@echo "实验:"
	@echo "  make jsq             运行 JSQ: SCENARIO=$(SCENARIO)"
	@echo "  make ga              运行 GA:  SCENARIO=$(SCENARIO)"
	@echo "  make ga GA_ARGS=\"--parallel-workers 6\"  指定 GA 并行评估进程数"
	@echo "  make evaluate        统计 JSQ vs GA: SCENARIO=$(SCENARIO)"
	@echo "  make run-scenarios   依次运行 current/planned/full 的 JSQ、GA 和评估"
	@echo "  make scenario-summary 汇总三阶段 JSQ/GA 指标表"
	@echo "  make run             执行 data -> train -> jsq -> ga -> evaluate"
	@echo "  make scenario        显示当前场景对应输出路径"
	@echo ""
	@echo "场景别名:"
	@echo "  make jsq-current | jsq-planned | jsq-full"
	@echo "  make ga-current  | ga-planned  | ga-full"
	@echo "  make evaluate-current | evaluate-planned | evaluate-full"
	@echo "  legacy: jsq-small/jsq-hk/jsq-default, ga-small/ga-hk/ga-default"
	@echo ""
	@echo "质量检查:"
	@echo "  make test            运行测试"
	@echo "  make format          格式化代码"
	@echo "  make lint            静态检查"
	@echo "  make clean           清理缓存文件"

check-scenario:
	@case "$(SCENARIO)" in \
		current|planned|full|small_demo|hk_scale|default) ;; \
		*) echo "不支持的 SCENARIO=$(SCENARIO)，可选: $(SCENARIOS)"; exit 2 ;; \
	esac

scenario: check-scenario
	@echo "SCENARIO=$(SCENARIO)"
	@echo "JSQ schedule: $(JSQ_SCHEDULE)"
	@echo "JSQ metrics:  $(JSQ_METRICS)"
	@echo "GA schedule:  $(GA_SCHEDULE)"
	@echo "GA metrics:   $(GA_METRICS)"
	@echo "Evaluation:   $(EVALUATION_OUTPUT)"

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

data: setup
	$(MAKE) data-full
	$(MAKE) data-small
	$(MAKE) data-derived
	$(MAKE) data-validate

data-small: setup
	$(PYTHON) scripts/collect_hk_gtfs.py \
		--max-routes $(SMALL_ROUTE_COUNT) \
		--max-trips $(SMALL_TRIP_LIMIT) \
		--output $(TRIPS_SMALL)

data-full: setup
	$(PYTHON) scripts/collect_hk_gtfs.py \
		--max-routes $(FULL_ROUTE_COUNT) \
		--max-trips $(FULL_TRIP_LIMIT) \
		--output $(TRIPS_FULL)

data-derived: setup
	$(PYTHON) scripts/build_project_datasets.py

data-validate:
	$(PYTHON) scripts/validate_datasets.py

simulate:
	@test -f src/simulate_data.py || { echo "未找到 src/simulate_data.py"; exit 1; }
	$(PYTHON) -m src.simulate_data

train: setup
	$(PYTHON) -m src.energy_model

jsq: check-scenario setup
	$(PYTHON) -m src.baseline_jsq --scenario $(SCENARIO)

jsq-current: SCENARIO=current
jsq-current: jsq

jsq-planned: SCENARIO=planned
jsq-planned: jsq

jsq-full: SCENARIO=full
jsq-full: jsq

jsq-small: SCENARIO=small_demo
jsq-small: jsq

jsq-hk: SCENARIO=hk_scale
jsq-hk: jsq

jsq-default: SCENARIO=default
jsq-default: jsq

ga: check-scenario setup
	$(PYTHON) -m src.ga_optimizer --scenario $(SCENARIO) $(GA_ARGS)

ga-current: SCENARIO=current
ga-current: ga

ga-planned: SCENARIO=planned
ga-planned: ga

ga-full: SCENARIO=full
ga-full: ga

ga-small: SCENARIO=small_demo
ga-small: ga

ga-hk: SCENARIO=hk_scale
ga-hk: ga

ga-default: SCENARIO=default
ga-default: ga

evaluate: check-scenario setup
	$(PYTHON) -m src.evaluation \
		--jsq-metrics $(JSQ_METRICS) \
		--ga-metrics $(GA_METRICS) \
		--jsq-schedule $(JSQ_SCHEDULE) \
		--ga-schedule $(GA_SCHEDULE) \
		--output $(EVALUATION_OUTPUT)

evaluate-current: SCENARIO=current
evaluate-current: evaluate

evaluate-planned: SCENARIO=planned
evaluate-planned: evaluate

evaluate-full: SCENARIO=full
evaluate-full: evaluate

evaluate-small: SCENARIO=small_demo
evaluate-small: evaluate

evaluate-hk: SCENARIO=hk_scale
evaluate-hk: evaluate

evaluate-default: SCENARIO=default
evaluate-default: evaluate

scenario-summary: setup
	$(PYTHON) -m src.scenario_comparison \
		--output-csv $(SCENARIO_SUMMARY_CSV) \
		--output-json $(SCENARIO_SUMMARY_JSON)

run-scenarios:
	@for scenario in $(PRIMARY_SCENARIOS); do \
		$(MAKE) jsq SCENARIO=$$scenario; \
		$(MAKE) ga SCENARIO=$$scenario; \
		$(MAKE) evaluate SCENARIO=$$scenario; \
	done
	$(MAKE) scenario-summary

figures:
	@test -f src/visualization.py || { echo "未找到 src/visualization.py"; exit 1; }
	$(PYTHON) -m src.visualization

run: experiment

experiment:
	$(MAKE) data
	$(MAKE) train
	$(MAKE) jsq
	$(MAKE) ga
	$(MAKE) evaluate

test:
	@if find tests -type f \( -name "test_*.py" -o -name "*_test.py" \) | grep -q .; then \
		$(RUN) pytest -q; \
	else \
		echo "未发现测试文件，跳过 pytest。"; \
	fi

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
