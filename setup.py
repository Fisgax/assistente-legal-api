from setuptools import setup

setup(
    name="assistente-legal-api",
    version="1.0.0",
    install_requires=open('requirements.txt').read().splitlines(),
)
