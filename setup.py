"""Setup configuration for praxis-sentence-transformer package."""
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_package_data=True,
    ) 