.PHONY: help install test test-cov test-fast lint format typecheck check clean benchmark

# 默认目标
help:
	@echo "Diffusion RAG — Makefile"
	@echo ""
	@echo "  make install      安装依赖"
	@echo "  make test         运行全部单元测试"
	@echo "  make test-cov     运行测试 + 覆盖率报告"
	@echo "  make test-fast    跳过编码器测试 (无需下载模型)"
	@echo "  make lint         代码格式检查 (black + isort)"
	@echo "  make format       自动格式化代码"
	@echo "  make typecheck    类型检查 (mypy --strict)"
	@echo "  make check        全部静态检查 (lint + typecheck)"
	@echo "  make benchmark    运行基线评测 (NFCorpus)"
	@echo "  make benchmark-ms 运行基线评测 (MS-MARCO)"
	@echo "  make clean        清理缓存文件"
	@echo "  make all          安装 → 检查 → 测试"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -k "not test_encode"

lint:
	black . --line-length 100 --check
	isort . --check-only

format:
	black . --line-length 100
	isort .

typecheck:
	mypy src/ --strict

check: lint typecheck
	@echo "All static checks passed."

benchmark:
	python -m src.baseline.benchmark --dataset nfcorpus

benchmark-ms:
	python -m src.baseline.benchmark --dataset msmarco

clean:
	python -c "import shutil, pathlib; \
		[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; \
		[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.mypy_cache')]; \
		[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; \
		[pathlib.Path(p).unlink(missing_ok=True) for p in pathlib.Path('.').rglob('.coverage')]"
	@echo "Cache files cleaned."

all: install check test-cov
	@echo ""
	@echo "Phase 1 baseline pipeline — all checks passed."
