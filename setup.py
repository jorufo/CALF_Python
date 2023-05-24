import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calfpy",
    version="1.2",
    author="John Ford, Clark Jeffries, Diana Perkins",
    author_email="JoRuFo@gmail.com",
    description="Contains greedy algorithms for coarse approximation linear functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jorufo/CALF_Python",
    project_urls={
        "Bug Tracker": "https://github.com/jorufo/CALF_Python/issues",
    },
    install_requires = ['numpy>=1.20.2,<=1.23.0', 'pandas>=1.2.4,<=1.5.2', 'plotnine==0.10.0', 'scipy>=1.6.3,<=1.9.3'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    py_modules=['methods']
)