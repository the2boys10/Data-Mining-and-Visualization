Aim:
To test various variations of normalisation as well as implement a multi-layer perceptron.
Requirements:
Python version 3.6.3, it is also required that you either change the python code to the path where the test.data and train.data are stored or you can instead run the program within a folder containing these two files. Finally the program requires that matplotlib be installed to output graphs this should be a standard library for anaconda installations. If graphs are not needed or you fail to install matplotlib I have added a separate version of the code that should work without the package.

To clarify the 4 separate version have the following requirements:
- Assignment1.py (Used in writeup) : numpy(including np.float128), matPlotLib, Python 3.6.3, train.data and test.data in same folder.

- Assignment1WithoutMatPlotLib.py : numpy(including np.float128), Python 3.6.3, train.data and test.data in same folder.

- Assignment1withoutfloat128.py : numpy, Python 3.6.3, matPlotLib, train.data and test.data in same folder.

- Assignment1WithoutMatPlotLibAndFloat128.py : numpy, Python 3.6.3, train.data and test.data in same folder.

Within the program you will be able to change 3 settings these are
	
	shuffleVal, this can be set to 1 if we would like to shuffle the values in train/test.data, if we would like use the data unshuffled then simply set it to 0.
	
	howManyTests, this can be set to any positive real number and is the amount of times we would like to create a empty network using our train.data and test it on our test.data, this will very for each test if shuffleVal is 1, however if shuffleVal is 0 will return the same result for each run.
	
	iterations, this can be set to any positive real number and controls how many times we iterate through our training data for each run.

When the program is run it should first list all binary results including the accuracy, false positives and correct predictions. After these binary results the program will output the results for each non-binary case in order of regularisation terms starting from 0.