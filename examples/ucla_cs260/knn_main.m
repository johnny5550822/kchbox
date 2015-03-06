%% Created by King Chung Ho (Johnny Ho) in 030515
% For UCLA CS260 project (Winter 2015)
% Objective: Use kchbox-knn to build a model for prediction
% Feature: 2 features (Diastolic and systolic) along different time
% Number of training data: 39 patients, 20 with label 0, and 19 with label
% 1

%% Step 1 Creating training data (random) for 2 clusters using gaussian distribution.
% Assuming there are two clusters with two dimenions

clear all;
clc;

% parameters
num_data = 20;
train_label = zeros(1,num_data*2);

% ###### cluster 1
% Define cluster
mean_1 = 2;
sigma_1 = 0.9;
x1 = normrnd(mean_1,sigma_1,[1 num_data]);
y1 = normrnd(mean_1,sigma_1,[1 num_data]);
cluster_1 = [x1 ; y1]';
% Define label
train_label(1:num_data) = 1;

% ##### cluster 2
% Define cluster
mean_2 = 5;
sigma_2 = 0.9;
x2 = normrnd(mean_2,sigma_2,[1 num_data]);
y2 = normrnd(mean_2,sigma_2,[1 num_data]);
cluster_2 = [x2 ; y2]';
% Define label
train_label(num_data+1:end) = 2;

% Plot 
plot(cluster_1(:,1),cluster_1(:,2),'bo',cluster_2(:,1),cluster_2(:,2),'rx');

%join two clusters
train_d = [cluster_1; cluster_2];
%% Step 2 Define k
k = 5;

%% Step 3 & Step 4
clc;
test_d = [5.5 2.5]; % test data
test_label = kchbox_knn(k,test_d,train_d,train_label);

%% plot the test point, assuming binary classification
hold on
plot(cluster_1(:,1),cluster_1(:,2),'bo',cluster_2(:,1),cluster_2(:,2),'rx');
if (test_label==1)
   h = plot(test_d(1),test_d(2),'b+');
else
   h = plot(test_d(1),test_d(2),'r+');    
end
set(h,'linewidth',3);
hold off

%% DONE!

