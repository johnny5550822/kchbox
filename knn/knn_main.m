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

%% Step 1 Creating training data (random) for 2 clusters using gaussian distribution.
% Assuming there are two clusters with two dimenions

clear all;
clc;

% parameters
num_data = 20;
label = zeros(1,num_data*2);

% ###### cluster 1
% Define cluster
mean_1 = 2;
sigma_1 = 0.9;
x1 = normrnd(mean_1,sigma_1,[1 num_data]);
y1 = normrnd(mean_1,sigma_1,[1 num_data]);
cluster_1 = [x1 ; y1]';
% Define label
label(1:num_data) = 1;

% ##### cluster 2
% Define cluster
mean_2 = 5;
sigma_2 = 0.9;
x2 = normrnd(mean_2,sigma_2,[1 num_data]);
y2 = normrnd(mean_2,sigma_2,[1 num_data]);
cluster_2 = [x2 ; y2]';
% Define label
label(num_data+1:end) = 2;

% Plot 
plot(cluster_1(:,1),cluster_1(:,2),'bo',cluster_2(:,1),cluster_2(:,2),'rx');

%join two clusters
j_cluster = [cluster_1 cluster_2];
%% Step 2 Define k
k = 4;

%% Step 3 Given a test data, find k closest neighbor.
clc;
test_data = [2.5 2.5];

% Calculate pair-wise distance between each point with the test data
pdist2(test_data,j_cluster,'euclidean');



