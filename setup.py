import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="predicting_APMALDI_response",
    author="Alberto Bailoni",
    author_email="alberto.bailoni@embl.de",
    description="Library for predicting AP-MALDI response for different matrices using a ML model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'predicting_APMALDI_response = predicting_APMALDI_response.__main__']
    }
)
