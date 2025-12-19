
import os
from setuptools import Extension, find_packages, setup



def get_requires():
    requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements, encoding="utf-8") as reqfile:
        return [str(r).strip() for r in reqfile]


setup(
    name="matcha-tts-develop",
    version="0.1.0",
    author="Paul-Marie Demars",
    author_email="paul-marie.demars@gadz.org",
    description="Un système de synthèse vocale (TTS) basé sur Matcha-TTS",
    long_description_content_type="text/markdown",
    install_requires=get_requires(),
    include_package_data=True,
    packages=find_packages()

)





