from os.path import join, dirname

from setuptools import setup, find_packages

# IMPORTANT: updates to the following must also be done in __init__.py.
__title__ = "datascience"
__version__ = "0.1.0"
__author__ = "Hugues Demers"
__email__ = "hdemers@gmail.com"
__copyright__ = "Hugues Demers"
__license__ = "MIT"

setup(
    name='datascience',
    version=__version__,
    description='Data science library and scripts',
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    author=__author__,
    author_email=__email__,
    license=__license__,
    packages=find_packages(exclude=('tests', 'docs', 'data')),
    #scripts=['bin/'],
    install_requires=[
    ],
)
