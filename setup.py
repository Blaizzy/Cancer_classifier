import setuptools

long_description=" This packeage was made for rapid prototyping and was optimized to work on Wisconsin Breast Cancer Database"
setuptools.setup(
	name="Cancer_classifier",
	version="0.0.1",
	author="Prince Canuma",
	author_email="princecanuminha@gmail.com",
	description="Classification of breast cancer",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/Blaizzy/Cancer_classifier/",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
