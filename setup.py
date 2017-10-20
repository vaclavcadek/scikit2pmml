# coding=utf-8
import os
from setuptools import setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('README.rst') as readme_file:
    long_description = readme_file.read()

setup(
    name='sklearn2pmml',
    version='0.0.1',
    packages=['sklearn2pmml'],
    include_package_data=True,
    license='MIT',
    description='Simple exporter of sklearn models into PMML.',
    long_description=long_description,
    url='https://github.com/vaclavcadek/sklearn2pmml',
    author='Václav Čadek',
    author_email='vaclavcadek@gmail.com',
    install_requires=[
        'numpy>=1.6.1',
        'SciPy>= 0.9',
        'scikit-learn>=0.17.1'
    ],
    tests_require=[
        'lxml',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
