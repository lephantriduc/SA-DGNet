from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        "torch==1.12.1",
        "pycox==0.2.3",
        "scikit-learn==1.2.2",
    ],
)
