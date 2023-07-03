from setuptools import setup, find_packages
import kernels

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='kernels',
    version=kernels.__version__,
    author='Parthe Pandit',
    author_email='parthe1292@gmail.com',
    description='Fast solver for Kernel Regression using GPUs with linear space and time complexity',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/parthe/pytorch_kernel_implementations/',
    project_urls = {
        "Bug Tracker": "https://github.com/parthe/pytorch_kernel_implementations//issues"
    },
    license='GNU General Public License v3.0',
    packages=find_packages(),
    install_requires=[],
)
