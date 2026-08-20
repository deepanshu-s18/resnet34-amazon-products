# Makefile for ResNet-34 + Grad-CAM experiment pipeline
# Usage: make <target>
# All experiments are config-driven — changing hyperparameters means editing a YAML, not code.

.PHONY: install test lint train-baseline train-resnet34 ablation evaluate grad-cam clean tensorboard

# ─── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -e .
	pip install -r requirements.txt

# ─── Quality ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# ─── Training ─────────────────────────────────────────────────────────────────
# E-001: BasicCNN baseline (validates training pipeline)
train-baseline:
	python scripts/train.py --config configs/cifar10_baseline.yaml

# E-002: ResNet-34 (main model, ~91% CIFAR-10 accuracy)
train-resnet34:
	python scripts/train.py --config configs/cifar10_resnet34.yaml

# E-003: ResNet-34 on Amazon Berkeley Objects
train-abo:
	python scripts/train.py --config configs/abo_resnet34_v1.yaml

# ─── Ablation studies ────────────────────────────────────────────────────────
# All ablations use identical hyperparameters; only one field differs from E-002
ablation-no-skip:
	python scripts/train.py --config configs/ablation_no_skip.yaml

ablation-label-smooth:
	python scripts/train.py --config configs/ablation_label_smooth.yaml

ablation-cosine:
	python scripts/train.py --config configs/ablation_cosine_lr.yaml

# Run all ablations sequentially (6-8 hours on 1 GPU)
ablation: train-baseline train-resnet34 ablation-no-skip ablation-label-smooth ablation-cosine

# ─── Evaluation ──────────────────────────────────────────────────────────────
evaluate:
	python scripts/evaluate.py \
		--checkpoint results/cifar10/cifar10_resnet34_E002_seed42/checkpoints/best.pt \
		--config configs/cifar10_resnet34.yaml \
		--compute-grad-cam \
		--output-dir results/evaluation/

# ─── Monitoring ──────────────────────────────────────────────────────────────
tensorboard:
	tensorboard --logdir results/

# ─── Dry runs (validate setup without full training — 2 batches only) ────────
dry-run-resnet34:
	python scripts/train.py --config configs/cifar10_resnet34.yaml --dry-run

dry-run-abo:
	python scripts/train.py --config configs/abo_resnet34_v1.yaml --dry-run

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name "*.egg-info" -exec rm -rf {} +
