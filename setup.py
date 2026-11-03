from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="transtimegrad",
    version="1.1.0",
    description="Transformer Based TimeGrad Model",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Nikita Bukhanchenko",
    author_email="nbukhanchenko@nes.ru",
    url="https://github.com/nbukhanchenko/transtimegrad",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=[
        "gluonts==0.16.3",
        "lightning==2.6.6",
        "diffusers==0.42.1",
        "torch==2.15.0",
        "numpy>=1.26.4,<3",
        "pandas>=2.2.3,<3",
        "scipy>=1.15.3,<2",
        "safetensors>=0.8.0,<1",
        "fsspec>=2026.6.0,<2027",
    ],
    extras_require={
        "test": ["pytest>=9.0.3,<10"],
        "examples": ["matplotlib>=3.10,<4"],
        "audit": ["pip-audit>=2.10,<3"],
    },
)
