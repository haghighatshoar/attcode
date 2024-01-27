from setuptools import setup

setup(
    name='atco',
    packages=['atco'],
    description='python module for designing channel codes based on attention mechanism used in LLMs',
    version='0.1',
    url='https://github.com/haghighatshoar/attcode',
    author='Saeid Haghighatshoar',
    author_email='haghighatshoar@gmail.com',
    download_url='https://github.com/haghighatshoar/attcode',
    keywords=[
        'Attention mechanism',
        'channel code'
        'error correction code',
        'iterative decoding',
    ],
    install_requires=[
        'numpy', 
        'torch',
    ]
)
