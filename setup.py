"""Install resnet-gradcam-amazon-products as an editable package.

After running `pip install -e .`, any script can:
    from src.models.resnet34 import ResNet34
    from src.explainability.gradcam import GradCAM
without sys.path manipulation.
"""
from setuptools import setup, find_packages

setup(
    name="resnet34_amazon_product",
    version="0.1.0",
    description="ResNet-34 from scratch + Grad-CAM for Amazon product image classification",
    packages=find_packages(where=".", exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
    ],
)
