% KNN on a test data
% Input: test_d(1xf), train_d(mxf), train_label(1xm), where f is the number
% of features, m is the number of training data.
% Output: prediction of the test data.

function prediction = kchbox_knn(k,test_d,train_d,train_label)
    % ########## Step 3 Given a test data, find k closest neighbor.
    % This part can be improved by using kd-tree. If your training dataset is
    % small, the basic implement is alright. If it is big, it will be much
    % better to use kd-tree.
    % ###########
    % Calculate pair-wise distance between each point with the test data
    D = pdist2(test_d,train_d,'euclidean');

    % find k nearest neighbors (To be specific, position in the array)
    % Sort the distance from small to large. The first k elements in pos is the
    % cloest neighbors
    [sorted_D,pos] = sort(D);
    neighbors_pos = pos(1:k);

    % ########## Step 4 Classify the test data by majority vote of the neighbors
    prediction = mode(train_label(neighbors_pos));


end




