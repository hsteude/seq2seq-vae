from setuptools import setup, find_packages

setup(
    name='seq2seq-vae',
    packages=find_packages(),
    install_requires=[
        'sklearn',
        'pandas',
        'numpy',
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
