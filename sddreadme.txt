# Generation - creates a "source" distribution of a package
python setup.py sdist

# Usage
1. Needs 64-bit python >= 3.6.1, java version "1.8.0_221"
2. Install virtual environment
	python -m venv mldev
	mldev\Scripts\activate.bat
3. Install sdd 
	pip install sdd-0.1.tar.gz
4. Install NLTK data.
   Run the python interpreter and type the commands
	python
	>>> import nltk
	>>> nltk.download()
	>>> exit()
5. Install spacy and download model
   pip install spacy
   python -m spacy download en_core_web_md
6. cd sdd
python test.py test
7. A file out_<YYYYMMDD>_<HHMM>.txt should be created in the current folder
	file, category
	test\Recipes-142-Barbecue.txt,none
	test\Recipes-Tea.txt,none