# _*_ coding: utf-8 _*_
"""
Time:     2022-05-04 1:10
Author:   Haolin Yan(XiDian University)
File:     setup.py.py
"""
from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mnn-meter',
    include_package_data=True,
    version="2.0.0",
    author="Haolin Yan",
    author_email='haolinyan_xdu@163.com',
    description="Tools for quickly building operator latency tables and for accurately predicting model latency (based on Pytorch and MNN)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3',
        # 省略一下
    ],
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "tabulate",
        "pyyaml",
    ],
    project_urls={
        'Source': 'https://github.com/makerlin1/MMT',
    }
)
