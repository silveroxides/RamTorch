from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ramtorch",
    version="1.8.0",
    author="Lodestone",
    author_email="lodestone.rock@gmail.com",
    description="RAM is All You Need",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/lodestone-rock/RamTorch",
    packages=find_packages(include=["ramtorch", "ramtorch.*"]),
    install_requires=["torch"],
    extras_require={
        "aimdo": ["comfy-aimdo>=0.4.5"],
        "uel": ["unifiedefficientloader>=0.5.4"],
    },
    py_modules=["ramtorch_aimdo_bootstrap"],
    entry_points={"console_scripts": ["ramtorch-aimdo=ramtorch_aimdo_bootstrap:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
