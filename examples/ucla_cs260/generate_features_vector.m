% Generate features vector from original data. There are several method of
% generating features. 

function fv = generate_features_vector(dia,systo)
    %##########Method 1: Simple concatanation############
    fv = [dia systo];

    %##########Method 2
    f1 = max(dia,[],2);
    f2 = min(dia, [],2);
    f3 = max(systo,[],2);
    f4 = min(systo,[],2);
    f5 = f1-f2;
    f6 = f3-f4;
    fv = [f1 f2 f3 f4 f5 f6];

    %###########MEthod 3
    diff = [systo - dia];
    ff1 = max(diff,[],2);
    ff2 = min(diff,[],2);
    ff3 = mean(diff,2);
    ff4 = std(diff,[],2);
    
    fv = [fv ff1 ff2 ff3 ff4];


end
