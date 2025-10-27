# =======================
# Settings
# =======================
PYTHON   ?= python3
VENV      := .venv
BIN       := $(VENV)/bin
PIP       := $(BIN)/pip
PY        := $(BIN)/python

ENTRY ?= main.py           # твой файл инференса
REQ_FILE ?= requirements.txt
YOLO_CACHE ?= .ultra_cache # кэш Ultralytics (веса YOLO)

ARGS ?=                     # прим.: make run ARGS="--input my_cars1.mp4"

.DEFAULT_GOAL := help

# =======================
# Help
# =======================
help:  ## Показать список команд
	@grep -E '^[.a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# =======================
# Env & deps (локально)
# =======================
venv: ## Создать venv
	$(PYTHON) -m venv $(VENV)

install: venv ## Установить зависимости из requirements.txt
	$(PIP) install -U pip
	@if [ -f $(REQ_FILE) ]; then $(PIP) install -r $(REQ_FILE); else echo "No $(REQ_FILE) found"; fi
	@mkdir -p $(YOLO_CACHE)

freeze: ## Зафиксировать версии в requirements.lock.txt
	$(PIP) freeze > requirements.lock.txt

# =======================
# Run (локально)
# =======================
run: ## Запустить инференс с визуализацией (окно cv2.imshow)
	@if [ -f $(ENTRY) ]; then SHOW_PREVIEW=1 YOLO_CONFIG_DIR=$(YOLO_CACHE) $(PY) $(ENTRY) $(ARGS); else echo "No $(ENTRY) found"; exit 1; fi

run-headless: ## Запустить без окна (только сохранение MP4)
	@if [ -f $(ENTRY) ]; then SHOW_PREVIEW=0 YOLO_CONFIG_DIR=$(YOLO_CACHE) $(PY) $(ENTRY) $(ARGS); else echo "No $(ENTRY) found"; exit 1; fi

# =======================
# Clean
# =======================
clean: ## Удалить временные файлы/кеши
	rm -rf dist build *.egg-info
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache|.ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage
