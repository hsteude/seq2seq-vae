from setuptools import setup, find_packages

setup(
    name='vae-ts-test',
    packages=find_packages(),
    install_requires=[
        'sklearn',
        'pandas',
        'numpy',
        'loguru',
        'torch',
        'pytorch_lightning',
        'matplotlib',
        'jupyter',
        'plotly',
        'interact'
        ],
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)
