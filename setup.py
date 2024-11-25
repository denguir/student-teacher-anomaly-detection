from setuptools import find_packages, setup

setup(
    name="anomaly_detection",
    version="0.1.0",
    packages=find_packages(),
    entry_points={"console_scripts": ["anomaly_detection=src.anomaly_detection:main"]},
)
