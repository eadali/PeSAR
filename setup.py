from setuptools import setup, find_packages

setup(
    name="PeSAR",
    version="0.1.0",
    description="PeSAR: Perception for Aerial Search and Rescue",
    long_description=open("README.md", encoding="utf-8").read() if __import__('os').path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Erkan Adali",
    author_email="erkanadali91@gmail.com",
    url="https://github.com/eadali/PeSAR",
    license="MIT",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python-headless",
        "ultralytics",
        "supervision",
        "sahi"
    ],
    keywords=[
        "aerial", "deep learning", "search and rescue", "camera", "sensor fusion"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
)