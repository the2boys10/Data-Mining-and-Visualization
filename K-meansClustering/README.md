Problem:
To create a k-means classifier and test various methods of regularisation and measuring techniques.
Requirements:
To run this python program, simply put the datasets ("animals",'countries","fruits","veggies") in the same folder as Kmeans2.py
additonally you will need to have both DistanceMeasures.py, MeasurePerformanceOfFinalClusters.py and graphing.py in the same
folder as well.

The requirements to run the Python are:
- Numpy library
- Matlab library
- Math library
- pathlib library (Used to write to file)
- copy library

Using python version:
- 3.6.3

The program will ask you how many run's you would like to do for every k, please choose a number
after which it will start to process all k-means settings outputting the graph onscreen as well as outputting them to a folder
The program will also output the metrics making up each graph, in "overallResults", in # seperated context, where each column is
an array storing the results for every k value:
typeOfSetting # average f-score # average precision # average recall # SD f-score # SD precision # SD recall # Max f-score # Max precision # Max Recall

The report was made with 1000 runs however this may take a large amount of time.