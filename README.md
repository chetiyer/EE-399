# EE-399
Compilation of homeworks for Machine Learning for Electrical Engineers Spring 2023
Curve fitting for Regression
Chetana Iyer 

Regression attempts to estimate the relationship among different variables in a system loosely based on the relationship between the dependent variables Y, the independent variables X, and an unknown parameter Beta. The parameters, Beta are calculated by optimizing the “goodness of fit” or how well the function matches to the given data. In this assignment, curve fitting was examined as a specific instance of regression. 

Sec. I. Introduction and Overview

Regression is important in the field of machine learning, and is based on the concept of constructing an objective function demonstrating the relationship between the independent (X)  and dependent variables(Y)  and parameters Beta, and optimizing the parameters by minimizing the error between the model’s vs. true values. There are multiple approaches to minimize error, but the three standard possibilities are Maximum Error, Mean Absolute Error, and the Least-squares error. This assignment uses the error metric of least-squares error in optimizing the parameters. 

Sec. II. Theoretical Background

This assignment specifically looked at nonlinear regression, which requires a more general mathematical framework for curve fitting. The given objective function was 

and the error metric used was the least-squares error. 

The assignment called for first finding the minimum least squares error and determining the parameters A,B,C,D from the given data, fixing two and sweeping through the other two parameters and generating a respective 2D loss landscape, and finally using various combinations of the data points as a data points and training points to fit a line, parabola and a 19th degree polynomial and compare the respective errors between the test versus training data for each fit. 

Sec. III. Algorithm Implementation and Development 

Finding the optimized parameters & error value from given data: First defined a function that calculated and returned the least squares error of the given objective function, then defined a guess initial guess value for the parameters, and used both as the first two parameters for the scipy.optimize.minimize function to calculate the optimized values of the parameters. Then, used .x and .fun to respectively retrieve the calculated optimized of the parameters and error. 

Sweeping through 2 of the 4 parameters and creating a loss landscape
		For each of the 2 parameters selected to be swept through, created a grid of values around the optimized value of those parameters, and used the meshgrid function of numpy. Then computed the Least squared error for each combination/sweeping of the two varying parameters, while keeping the other remaining two parameters to the optimized value, and creating a final grid reflecting that of the error of the swept values. Used plt.pcolor to visualize the loss landscape.

Training/Testing: Used np.polyfit to fit with appropriate parameters reflecting that of a line, parabola and 19th degree polynomial to calculate coefficients of the polynomial. Then used that value to fit based off the training and testing data and calculate the error of the respective method by subtracting the calculated value to the given true Y value (Y_train or Y_test) 


Sec. IV. Computational Results

Part A: Minimum Error: 1.56, Optimized Parameter values: 2.17, 0.909, 0.732, 31.45

Part B: (look at code for images, did not know how to include here) 

Sec. V. Summary and Conclusions
	      We can conclude that the loss landscapes reflect values around the same range of the optimized parameter values, with one exception being the B parameter having an additional value at -.91. My other takeaways include the impact of sectioning out different parts of a data set for training and testing, and how it is important to find a balance between overfitting and completely reducing the error on the training data set to actually having a working model that might have a slightly higher error - but will still work on all testing data. 







