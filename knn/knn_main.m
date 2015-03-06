%% Created by King Chung Ho (Johnny Ho) in 030515


%*****************************
% K-nearest neighbor (KNN) is an algorithm which classify an incoming new
% sample by looking into k nearest neighbors of the new sample. For example,
% in a problem of classifying whether an object is an apple or not based on 
% texture and color, if more % than 2 % out of 5 neightbors of the object is 
% apple, then the object will be classified as an apple. 

%****************************
% The flow of KNN is below:

% Step 1: Obtain training data (maybe do some preprocessing)
% Step 2: Define k. k is arbitrary. It can be tuned by cross-validation.
% Step 3: For a new sample, find the closest k-neighbors based on a
% distance measurement, e,g, a common one is nucliean distance.
% Step 4: Classify the new sample by getting the majority vote of
% k-neighbors.

% Note: in step 3, simple straight forward way to find the cloest neighbors
% is by calculating every pair of possible distance. However, a faster way
% to is use a data structure called kd-tree, speed: O(nlogn). In this
% example, I will demonstrate the straight forward way. It should not be
% difficult to extend the code to use kd-tre.

%% 