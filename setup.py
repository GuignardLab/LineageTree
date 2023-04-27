from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="LineageTree",
    version="1.1.0",
    description="Lineage tree structure for TGMM algorithm",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/leoguignard/TGMMlibraries",
    author="Leo Guignard",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["LineageTree"],
    package_dir={"": "src"},
    install_requires=["scipy>=1.9", "numpy>=1.23"],
    extras_require={"svg": ["svgwrite", "matplotlib"]},
)
