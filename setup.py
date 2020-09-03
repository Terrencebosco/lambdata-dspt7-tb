# setup.py file

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Terrence_Bosco_utilities", # the name that you will install via pip
    version="1.0",
    author="Terrence Bosco",
    author_email="Terrencebosco@gmail.com",
    description="a few helper functions for assignment 1 unit 3 mod 1",
    long_description=long_description,
    long_description_content_type="text/markdown", # required if using a md file for long desc
    #license="MIT",
    url="https://github.com/Terrencebosco/lambdata-dspt7-tb",
    #keywords="",
    packages=find_packages() # ["my_lambdata"]
)