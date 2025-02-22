from setuptools import setup, find_packages

setup(
    name="PixelHop",
    version="0.1.0",
    description="A package for PixelHop with Saab transformation and shrinking operations.",
    author="Hong-En Chen",
    author_email="hongench@usc.edu",
    packages=find_packages(),
    install_requires=[
        "jax",
        "flax",
        "einops",
        "numpy",
    ],
)
