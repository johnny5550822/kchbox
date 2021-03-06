% Generate features vector from original data. There are several method of
% generating features. 

function fv = generate_features_vector(dia,systo)
    %##########Method 1: Simple concatanation############
    fv = [dia systo];

    %##########Method 2: Basic features
    f1 = max(dia,[],2);
    f2 = min(dia, [],2);
    f3 = max(systo,[],2);
    f4 = min(systo,[],2);
    f5 = f1-f2;
    f6 = f3-f4;
    f7 = mean(dia,2);
    f8 = mean(systo,2);
    f9 = std(dia,[],2);
    f10 = std(systo,[],2);
    
    %skewness
    f11 = (sum((dia-repmat(f7,[1 size(dia,2)])).^3,2)/size(dia,2))./(f9.^3);
    f12 = (sum((systo-repmat(f7,[1 size(systo,2)])).^3,2)/size(systo,2))./(f10.^3);
    
    %kurtosis: whether the data are peaked or flat relative to a normal
    %distribution
    f13 = (sum((dia-repmat(f7,[1 size(dia,2)])).^4,2)/size(dia,2))./(f9.^4);
    f14 = (sum((systo-repmat(f7,[1 size(systo,2)])).^4,2)/size(systo,2))./(f10.^4);
    
    % number of peaks
    [a,f15,f16,f17] = convertSeriesToBinary(dia);
    [b,f18,f19,f20] = convertSeriesToBinary(systo);
    
    %combine all feature
    fv = [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f15 f16 f18 f19];
    
    %fv = [f15 f16 f17 f18];
    %###########MEthod 3: Basic features from the difference of systolic
    %and diastolic pressure
    diff = [systo - dia];
    ff1 = max(diff,[],2);
    ff2 = min(diff,[],2);
    ff3 = mean(diff,2);
    ff4 = std(diff,[],2);
    [a,ff5,ff6] = convertSeriesToBinary(diff);
    
    % fv = [fv ff1 ff2 ff3 ff4 ff5 ff5];

    % normalize to range between 0 and 1
    fv = (fv - repmat(min(fv),[size(fv,1) 1]))./repmat((max(fv)-min(fv)),[size(fv,1) 1]);

    
    %##########method4: Fourier transform: transfer the time-series into
    %frequency
    % fv = convertToFreqFeatures(systo);
end
