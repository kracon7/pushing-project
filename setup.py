from setuptools import find_packages, setup

setup(
    name='taichi_pushing',
    version='0.1.0',
    description='A differentiable particle based rigid body physics engine in Taichi.',
    author='Jiacheng Yuan',
    author_email='yuanx320@umn.edu',
    platforms=['any'],
    install_requires=['taichi'],
    packages=find_packages(),
    python_requires=">=3.6",
)
