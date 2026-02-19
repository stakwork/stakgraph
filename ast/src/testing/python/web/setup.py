from setuptools import setup, find_packages

setup(
    name="stakgraph",
    version="0.1.0",
    description="Multi-framework web application example",
    author="StakGraph",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "flask",
        "uvicorn",
        "sqlalchemy",
        "pydantic",
        "django"
    ],
)
