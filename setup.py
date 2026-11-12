from setuptools import setup, find_packages

setup(
    name="surf-llm",
    version="0.1.0",
    description="Sparse Unlearning with Reasoning-Preserving Framework for Large Language Models",
    author="Krishnanand Tiwari",
    author_email="krishnanand.tiwari@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
