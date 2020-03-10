"""
A collection of functions to quickly split data and analyze it

"""
import setuptools

REQUIRED = [
    "numpy",
    "pandas",
    "scikit-learn"
]

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()
    setuptools.setup(
        name="class_test",
        version="1",
        author="evidencen",
        description="A collection of functions to quickly split data and analyze it",
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://lambdaschool.com/courses/data-science",
        packages=setuptools.find_packages(),
        python_requires=">=3.5",
        install_requires=REQUIRED,
        classifiers=["Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                     ]
    )
