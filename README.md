# CPSC310_Project

Our program is in the Andrews_Mulderink_CPSC310_Project.ipynb file. 
Open this file in jupyter notebook and select run all cells to use it.
If using more specific cells is desired, some guidelines:
The initial code cell does imports including the csv file we used, so it
must run before any other cells. The modified kNN cells must run 
sequentially since they use the one above them. Then kNN ensemble however
only requires the inital cell. 
Cells with primary functions, such as classifiers, will run those functions
after declaring them. Cells with helper functions will not. 

Organization:
The first markdown cell is an intro and the first code cell contains
imports and sets up the table and header.
The second code explores the dataset by creating the graphs
The following cell declares helper function get_random_attribute_subset
which is used for kNN testing different subsets of attributes and for
the ensemble of kNN classifiers. 
The following cell defines and runs a zeroR classifier over the table.
The following cell defines a kNN which tests different values of k, 
then runs this over the column displaying the different possible measurements.
The following cell defines a kNN which tests random attribute subsets.
This uses the k found in the previous cell.
The following cell defines two helper functions, which are used in the kNN
with randoms weights and in the ensemble.
The following cell defines a function to test kNN with random weights.
This uses the previously found best k and attribute subset.
The following cell creates and tests an ensemble classifier composed
of kNN classifiers with random k, attribute subsets, and weights.
The following cell tests logistic regression.
The final cell tests naive bayes. 