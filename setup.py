from setuptools import setup, find_packages

setup(
    name="transtimegrad",
    version="1.0",
    description="Transformer Based TimeGrad Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nikita Bukhanchenko",
    author_email="nbukhanchenko@nes.ru",
    url="https://github.com/nbukhanchenko/transtimegrad",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.8",
    install_requires=[
        "gluonts==0.14.3",
        "seaborn==0.13.1",
        "pandas==2.2.0",
        "lightning==2.1.3",
        "diffusers==0.25.1",
        "torchvision==0.17.2",
        "torch==2.2.2",
        "numpy==1.23.5",
        "holidays",
        "matplotlib",
        "protobuf~=3.20.3",
    ],
    tests_require=["flake8", "pytest"],
)
