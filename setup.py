import io

from setuptools import find_packages, setup

description = "Divide and Compute with Maximum Likelihood Tomography"

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()

qsplit_mlft_packages = ["qsplit_mlft"] + [
    "qsplit_mlft." + package for package in find_packages(where="qsplit_mlft")
]

# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

setup(
    name="QSPLIT-MLFT",
    version="0.0.1",
    url="https://github.com/Quantum-Software-Tools/QSPLIT-MLFT",
    author="Michael Perlin",
    author_email="mika.perlin@gmail.com",
    python_requires=(">=3.8.0"),
    install_requires=requirements,
    license="3-Clause BSD",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=qsplit_mlft_packages,
)