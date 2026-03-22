from setuptools import setup, find_packages

setup(
    name="rl-basics",
    version="0.1.0",
    description="Reinforcement Learning Basics - From Q-Learning to SAC",
    author="Carl-Creat",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.28.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
