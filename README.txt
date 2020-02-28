All code is contained in the Jupyter notebook Randomized_Optimization.ipynb

Code can be found here: https://github.com/luclement/ml-assignment2

Datasets:

1. Credit Card Default dataset is provided under datasets folder, if it does not work for some reason, it can be downloaded here: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/download
  a. Place under datasets/

Make sure you have Java 8 (jython doesn't work with later versions) and Python 3 installed.

Install:
1. Install jython: https://www.jython.org/installation.html (or "brew install jython" if on macOS)
2. Run "pip install -r requirements.txt" to install required dependencies
3. Navigate to mlrose/ and run "pip install ." to install the local package
4. Navigate to ABAGAIL/ and run "ant" to compile the custom ABAGAIL .jar file
5. Make sure you have Jupyter installed by running "jupyter --version" and you should see jupyter-notebook installed

Usage:
1. To start the notebook, run: jupyter notebook
2. RO experiments were generated using ABAGAIL:
    a. Navigate to ABAGAIL/jython
    b. Run all of the Python files to generate the results json file, e.g. jython fourpeaks.py (make sure ABAGAIL.jar is built)
2. Navigate to the Jupyter notebook and run from the top down

Code references:
1. ABAGAIL jython examples: https://github.com/pushkar/ABAGAIL/tree/master/jython
2. mlrose examples: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
3. utils from my assignment 1: https://github.com/luclement/ml-assignment1/blob/master/utils.py
4. In utils, credit to Sklearn for the learning curve plotting code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
5. In utils, credit to Sklearn for the validation curve plotting code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

Packages used:
1. scikit-learn: https://scikit-learn.org/stable/
2. jupyter: https://jupyter.org/
3. numpy: https://numpy.org/
4. matplotlib: https://matplotlib.org/i
5. mlrose: https://mlrose.readthedocs.io/en/stable/
6. mlrose_hiive: https://github.com/hiive/mlrose
7. ABAGAIL: https://github.com/pushkar/ABAGAIL
8. pandas: https://pandas.pydata.org/
