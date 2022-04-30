from setuptools import setup, find_packages

setup(
    name="calibrate",
    version="1.0",
    author="",
    description="For awesome calibration research",
    packages=find_packages(),
    python_requries=">=3.8",
    install_requires=[
        # Please install torch and torchvision libraries before running this script
        "torch",
        "torchvision>=0.8.2",
        "ipdb==0.13.9",
        "albumentations==1.1.0",
        "opencv-python==4.5.1.48",
        "hydra-core==1.1.2",
        "flake8==4.0.1",
        "wandb==0.12.14",
        "terminaltables==3.1.10",
        "matplotlib==3.5.1",
        "plotly==5.7.0",
        "pandas==1.4.2"
    ],
)
