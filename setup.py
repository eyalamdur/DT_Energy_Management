from setuptools import setup, find_packages

setup(
    name="decision_transformer",
    version="0.1",
    packages=find_packages(where="code"),
    package_dir={"": "code"},
)