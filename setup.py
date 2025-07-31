import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enhanced_cs_cv_caprecover",
    version="1.0.0",
    author="Senior Python Developer",
    author_email="example@enterprise-xr.com",
    description="Enhanced CS CV CapRecover Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enterprise-xr/enhanced_cs.CV_CapRecover",
    project_urls={
        "Bug Reports": "https://github.com/enterprise-xr/enhanced_cs.CV_CapRecover/issues",
        "Funding": "https://www.enterprise-xr.com/funding",
        "Say Thanks!": "https://github.com/enterprise-xr/enhanced_cs.CV_CapRecover/stargazers",
        "Contact": "https://www.enterprise-xr.com/contact",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch==1.13.1",
        "numpy==1.24.2",
        "pandas==1.5.2",
        "scipy==1.9.3",
        "scikit-image==0.19.3",
        "pillow==9.4.0",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    keywords="computer vision, xr, eye tracking",
)