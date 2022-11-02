import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pred_spot_intensity",
    author="Alberto Bailoni",
    author_email="alberto.bailoni@embl.de",
    description="Library for predicting matrix intensity for different matrices using a ML model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'pred_spot_intensity = pred_spot_intensity.__main__']
    }
)
