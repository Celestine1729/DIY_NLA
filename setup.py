from setuptools import setup, find_packages

setup(
    name="DIY_NLA",
    version="1.0.1",  # Will be automated later
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7",
)
