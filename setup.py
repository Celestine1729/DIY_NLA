from setuptools import setup, find_packages

setup(
    name="DNLAP",
    version="0.0.7",  # Will be automated later
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7",
)
