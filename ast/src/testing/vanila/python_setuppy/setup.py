from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mylib-core",
    version="0.1.0",
    author="Stakgraph Team",
    author_email="dev@example.com",
    description="A robust core library for data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/mylib",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pydantic>=1.10.0",
        "sqlalchemy>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "mylib-cli=mylib.core:main",
        ],
    },
)
