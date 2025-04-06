from setuptools import setup, find_packages

setup(
    name="AsymptoticRingModeSolver",
    version="0.0.1",
    author="Michael Sloan",
    author_email="michael.sloan@mail.utoronto.ca",
    description="Package to construct asymptotic-in and -out mode fields for general ring resonator structures.",
    long_description="Package to construct asymptotic-in and -out mode fields for a general ring resonator structure. This includes effects from higher order mode coupling in resonator bends and coupling region, as well as local variation in mode properties. Examples for generating mode properties using the open source mode solver 'Femwell' are included to provide a self-contained resource.",
    packages=find_packages(),
    install_requires=['femwell']
)