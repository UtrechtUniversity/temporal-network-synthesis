# based on https://github.com/pypa/sampleproject - MIT License

from setuptools import setup, find_packages

setup(
    name='synet',
    version='0.0.1',
    author='Raoul Schram',
    description='synet',
    long_description='more synet',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    python_requires='~=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'matplotlib',
        'networkx',
        'pandas',
    ]
)
