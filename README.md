Table of Content
1. Team details
2. Abstract
3. Algorithms Used
3.1 Regression
		3.1.1 Linear Regression
		3.1.2 Polynomial Linear Regression
		3.1.3 Decision Tree Regression
		3.1.4 Random Forest Regression
	3.2 Classification
		3.2.1 Random Forest Classification
		3.2.2 Decision Tree Classification
		3.2.3 K Nearest Neighbour
		3.2.4 Naïve Bayes
		3.2.5 SVM
4. Libraries Used
	4.1 Numpy
		4.1.1 History
		4.1.2 Traits
		4.1.3 Limitations
	4.2 Pandas
		4.1.1 History
	4.3 Scikit-learn
4.3.1 History
4.3.2 Implementation
5. Function Used
	5.1 sklearn.preprocessing
		5.1.1 sklearn.preprocessing.StandardScaler
		5.1.2 sklearn.preprocessing.MinMaxScaler
		5.1.3 sklearn.preprocessing.Normalizer
		5.1.4  sklearn.preprocessing.PolynomialFeatures
	5.2 sklearn.linear_model
		5.2.1 sklearn.linear_model.LinearRegression
		5.2.2 sklearn.linear_model.LogisticRegression
	5.3 sklearn.cross_validation.train_test_split
	5.4 sklearn.tree.DecisionTreeRegressor
	5.5 sklearn.ensemble.RandomForestRegressor

	

6. Code Description

7. Screenshots of Application

8. Code

9. Summary
Chapter – 1
Abstract

"Generally to implement Machine Learning, we have to have a basic understanding of a programming language (like python or R) and some packages (like numpy, pandas, sklearn etc) which are necessary for machine learning. Also Machine learning models requires computation power to run the algorithms. Our target is to eliminate all the above mentioned hurdles in the path to start ML."

"Our application “MoTE” allows the user to perform all the tasks from data processing to model selection without writing a single line of code. With the help of a user friendly interface the application takes the dataset as input, do all the computation and finally displays the score for the model. "


















Chapter – 2
Algorithms Used

2.1. Regression
In statistical modelling, regression analysis is a set of statistical processes for estimating the relationships among variables. It includes many techniques for modelling and analysing several variables, when the focus is on the relationship between a dependent variable (or ‘label’) and one or more independent variable (or 'features'). More specifically, regression analysis helps one understand how the typical value of the dependent variable (or 'criterion variable') changes when any one of the independent variables is varied, while the other independent variables are held fixed.
2.1.1 Linear Regression
Linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression. This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.
 

2.1.2 Polynomial Linear Regression
It uses a non-linear function. It extends the linear model by adding extra predictors obtained by raising each of the original predictors to a power. It is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as a nth degree polynomial in x.
 
2.1.3 Decision Tree Regression
It builds regression models in the form of a tree structure.It breaks down a dataset into smaller and smaller subsets while at the same time a associated decision tree is incrementally developed.The final result is a tree with decision nodes and leaf nodes. 
 

2.1.4 Random Forest Regression
Random Forest Regression is ensemble algorithm, creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.
2.2 Classification
Classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. Examples are assigning a diagnosis to a given patient based on observed characteristics of the patient (gender, blood pressure, presence or absence of certain symptoms, etc.). Classification is an example of pattern recognition.
2.2.1 Random Forest Classification
Random Forest Classifier is ensemble algorithm, creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.
 
2.2.2 Decision Tree Classification
In decision tree classification, it used to separate the dataset into classes belonging to the responses variable usually (1 or 0).


2.2.3 K Nearest Neighbour
When KNN is used for classification, the output can be calculated as the class with the highest frequency from the K-most similar instances.Each instances in essence votes for their class and the class with the most votes is taken as prediction.
 
2.2.4 Naïve Bayes
Naïve Bayes also known as Naïve Bayes Classifier are the classifiers with the assumption that faeatures are statistically independent of one another. Unlike many other classifiers which assumes that, for a given class there will be  some correlation between features, Naïve bayes explicitly models that features as conditionally independent given the class.
 
2.2.5 SVM
A SVM is a discriminative classifier formally defined by a separating hyperplane .In other words , given labelled training data (supervised learning ), the algorithm outputs an optimal hyperplane which categorizes new examples.In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.
 













Chapter – 3
Libraries Used

3.1 Numpy
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-levelmathematical functions to operate on these arrays. The ancestor of NumPy, Numeric, was originally created by Jim Huguninwith contributions from several other developers. In 2005, Travis Oliphant created NumPy by incorporating features of the competing Numarray into Numeric, with extensive modifications. NumPy is open-source software and has many contributors.

3.1.1. History
The Python programming language was not initially designed for numerical computing, but attracted the attention of the scientific and engineering community early on, so that a special interest group called matrix-sig was founded in 1995 with the aim of defining an array computing package. Among its members was Python designer and maintainer Guido van Rossum, who implemented extensions to Python's syntax (in particular the indexing syntax) to make array computing easier. There was a desire to get Numeric into the Python standard library, but Guido van Rossum decided that the code was not maintainable in its state then. In early 2005, NumPy developer Travis Oliphant wanted to unify the community around a single array package and ported Numarray's features to Numeric, releasing the result as NumPy 1.0 in 2006.[6] This new project was part of SciPy. To avoid installing the large SciPy package just to get an array object, this new package was separated and called NumPy. Support for Python 3 was added in 2011 with NumPy version 1.5.0.In 2011, PyPy started development on an implementation of the NumPy API for PyPy.[13] It is not yet fully compatible with NumPy.[14]
3.1.2. Traits
NumPy targets the CPython reference implementation of Python, which is a non-optimizing bytecode interpreter. Mathematical algorithms written for this version of Python often run much slower than compiled equivalents. NumPy addresses the slowness problem partly by providing multidimensional arrays and functions and operators that operate efficiently on arrays, requiring rewriting some code, mostly inner loops using NumPy.
Using NumPy in Python gives functionality comparable to MATLAB since they are both interpreted,[15] and they both allow the user to write fast programs as long as most operations work on arrays or matrices instead of scalars. In comparison, MATLAB boasts a large number of additional toolboxes, notably Simulink, whereas NumPy is intrinsically integrated with Python, a more modern and complete programming language. Moreover, complementary Python packages are available; SciPy is a library that adds more MATLAB-like functionality and Matplotlib is a plotting package that provides MATLAB-like plotting functionality. Internally, both MATLAB and NumPy rely on BLAS and LAPACK for efficient linear algebra computations.
Python bindings of the widely used computer vision library OpenCV utilize NumPy arrays to store and operate on data. Since images with multiple channels are simply represented as three-dimensional arrays, indexing, slicing or masking with other arrays are very efficient ways to access specific pixels of an image. The NumPy array as universal data structure in OpenCV for images, extracted feature points, filter kernels and many more vastly simplifies the programming workflow and debugging.
The core functionality of NumPy is its "ndarray", for n-dimensional array, data structure. These arrays are strided views on memory. In contrast to Python's built-in list data structure (which, despite the name, is a dynamic array), these arrays are homogeneously typed: all elements of a single array must be of the same type.
Such arrays can also be views into memory buffers allocated by C/C++, Cython, and Fortran extensions to the CPython interpreter without the need to copy data around, giving a degree of compatibility with existing numerical libraries. This functionality is exploited by the SciPy package, which wraps a number of such libraries (notably BLAS and LAPACK). NumPy has built-in support for memory-mapped ndarrays.
3.1.3. Limitations
Inserting or appending entries to an array is not as trivially possible as it is with Python's lists. The np.pad(...) routine to extend arrays actually creates new arrays of the desired shape and padding values, copies the given array into the new one and returns it. NumPy's np.concatenate([a1,a2]) operation does not actually link the two arrays but returns a new one, filled with the entries from both given arrays in sequence. Reshaping the dimensionality of an array with np.reshape(...) is only possible as long as the number of elements in the array does not change. These circumstances originate from the fact that NumPy's arrays must be views on contiguous memory buffers. A replacement package called Blaze attempts to overcome this limitation. 
Algorithms that are not expressible as a vectorized operation will typically run slowly because they must be implemented in "pure Python", while vectorization may increase memory complexity of some operations from constant to linear, because temporary arrays must be created that are as large as the inputs. Runtime compilation of numerical code has been implemented by several groups to avoid these problems; open source solutions that interoperate with NumPy include scipy.weave, numexpr[17] and Numba.[18] Cython is a static-compiling alternative to these.
3.2 Pandas
In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license. The name is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.
3.1.1 History
Developer Wes McKinney started working on pandas in 2008 while at AQR Capital Management out of the need for a high performance, flexible tool to perform quantitative analysis on financial data. Before leaving AQR he was able to convince management to allow him to open source the library. Another AQR employee, Chang She, joined the effort in 2012 as the second major contributor to the library. In 2015, pandas signed on as a fiscally sponsored project of NumFOCUS, a 501(c)(3) nonprofit charity in the United States.
3.3 Scikit-learn
Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
3.3.1 History
Scikit-learn was initially developed by David Cournapeau as a Google summer of code project in 2007. Later Matthieu Brucher joined the project and started to use it as a part of his thesis work. In 2010 INRIA, the French Institute for Research in Computer Science and Automation, got involved and the first public release (v0.1 beta) was published in late January 2010.
3.3.2 Implementation
Scikit-learn is largely written in Python, with some core algorithms written in Cython to achieve performance. Support vector machines are implemented by a Cython wrapper around LIBSVM; logistic regression and linear support vector machines by a similar wrapper around LIBLINEAR.













Chapter – 5
Function Used

5.1 sklearn.preprocessing
5.1.1 sklearn.preprocessing.StandardScaler

class sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
Parameter :	copy : boolean, optional, default True
		with_mean : boolean, True by default

5.1.2 sklearn.preprocessing.MinMaxScaler
class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
Parameters:	feature_range : tuple (min, max), default=(0, 1)
		copy : boolean, optional, default True

5.1.3 sklearn.preprocessing.Normalizer
class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
Parameters:	feature_range : tuple (min, max), default=(0, 1)
		copy : boolean, optional, default True

5.1.4  sklearn.preprocessing.PolynomialFeatures
class sklearn.preprocessing.Normalizer(norm=’l2’, copy=True
Parameters:	norm : ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default)
		copy : boolean, optional, default True

5.2 sklearn.linear_model
5.2.1 sklearn.linear_model.LinearRegression
class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
Parameters:	fit_intercept : boolean, optional, default True
		copy_X : boolean, optional, default True
		normalize : boolean, optional, default False
5.2.2 sklearn.linear_model.LogisticRegression
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
Parameters:	penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
		C : float, default: 1.0




5.3 sklearn.cross_validation.train_test_split
sklearn.cross_validation.train_test_split(*arrays, **options)
Parameters:	random_state : int or RandomState
		test_size : float, int, or None (default is None)

5.4 sklearn.tree.DecisionTreeRegressor

class sklearn.tree.DecisionTreeRegressor(criterion=’mse’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
Parameters:	max_depth : int or None, optional (default=None)
		random_state : int, RandomState instance or None, optional (default=None)

5.5 sklearn.ensemble.RandomForestRegressor

class sklearn.ensemble.RandomForestRegressor(n_estimators=10, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
Parameters:	max_features : int, float, string or None, optional (default=”auto”)
		max_depth : integer or None, optional (default=None)




























Chapter – 6
Codes Description
 
 
Figure 6.1
•	In figure 6.1, we are importing the pandas library to be used for accessed data from files. In lines 2, 4 and 5 we are importing different regression libraries. In lines 8 to 12 we are importing different classification libraries. 

 
Figure 6.2 
 
Figure 6.3
•	In figure 6.2, we declare a function name reg. In this we first take the dataset and store it in variable “dataset”. Then we divided the dataset into set of features named “x” and set of labels names “y”. After that we check what type of scaling method is opted by the user and accordingly we apply in on the features. Then we check which model has been selected by the user and then we call the function having the program for the required model. The function returns a score which is then returned by the “Reg” function. The same thing happens in Figure 6.3, but for the function “classify” that stores all the classifiers.

 
•	In the function “lin”, first the regressor object is created using class Linear Regression. The model is trained over training data “xtrain” and “ytrain”. The trained regressor is then returned from the function.

 
•	In the function “poly”, first an object of Polynomial Feature is created named poly. “poly” object is then trained over and tranform the feature set, the resultant set consists of a set having degree of x from 0 to 3. Then the regressor object is created using class Linear Regression. The model is trained over training data “xtrain” and “ytrain”. The trained regressor is then returned from the function.


 
•	In the function “decTree”, first the regressor object is created using class Decision Tree Regressor. The model is trained over training data “xtrain” and “ytrain”. The trained regressor is then returned from the function.

 
•	In the function “randFor”, first the regressor object is created using class Random Forest Regressor. The model is trained over training data “xtrain” and “ytrain”. The trained regressor is then returned from the function.




 
•	In the function “logReg”, first the regressor object is created using class Logistic Regression. The model is trained over training data “xtrain” and “ytrain”. The trained regressor is then returned from the function.

 
•	In the function “decTree”, first the classifier object is created using class Decision Tree Classifier. The model is trained over training data “xtrain” and “ytrain”. The trained classifier is then returned from the function.
 
•	In the function “randFor”, first the classifier object is created using class Random Forest Classifier. The model is trained over training data “xtrain” and “ytrain”. The trained classifier is then returned from the function.

 
•	In the function “svClass”, first the classifier object is created using class SVC Classifier. The model is trained over training data “xtrain” and “ytrain”. The trained classifier is then returned from the function.


 
•	In the function “naivBay”, first the classifier object is created using class Naïve Bayes Classifier. The model is trained over training data “xtrain” and “ytrain”. The trained classifier is then returned from the function.

 
•	In the function “knClass”, first the classifier object is created using class KNeighbors Classifier. The model is trained over training data “xtrain” and “ytrain”. The trained classifier is then returned from the function.












Chapter – 7
Screenshots of the Application

 
Figure 1. Upload a Dataset
 
Figure 2. Selecting Useful features and Labels
 
Figure 3. Data Processing Step  
Figure 4. Model Selecting
 
Figure 5. Result Display















Chapter – 8
Summary

"The main idea of our application “MoTE” allows the user to perform all the tasks from data processing to model selection without writing a single line of code. With the help of a user friendly interface the application takes the dataset as input, do all the computation and finally displays the score for the model. Generally to implement Machine Learning, we have to have a basic understanding of a programming language (like python or R) and some packages (like numpy, pandas, sklearn etc) which are necessary for machine learning. Also Machine learning models requires computation power to run the algorithms. Our target is to eliminate all the above mentioned hurdles in the path to start ML. Our application eliminates all the difficulties that a beginner faces if persons knows how ML works then person can perform all the tasks and learn about the dataset more efficiently and able to choose the best algorithms for that particular problem. It increases efficiency and conserve the time of the user. It also clears the concept of user about the dataset. There will be no need to code from scratch if the user knows the basic of machine learning then this application can be easily used on datasets and get the required Results in less amount of time."




