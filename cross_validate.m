function error=cross_validate(model,X,Y,k_folds)
%cross_validate This Function returns the error as measured by error_metric
%for a given predict_labels function for a given X and Y dataset and given
%k number of folds.
%   
%   Input Examples:
%       -model=@predict_labels
%       -X=train_inputs
%       -Y=train_labels
%       -k_folds=10

% Create cross validation folds
[n_observations,~]=size(Y);
cv_indices  = crossvalind('KFold',n_observations,k_folds);

%Initialize Error Sum Vector
error_sum=0;

for fold = 1:k_folds
    
    %Redefine test and train matrices
    test=cv_indices == fold;
    train=~test;
    X_train = X(train,:);
    X_test = X(test,:);
    Y_train = Y(train,:);
    Y_test = Y(test,:);
    
    [pred_labels]=model(X_train,Y_train,X_test);
    
    %Add the Error for this fold of cv to the Error Total
    error_sum=error_sum+error_metric(pred_labels,Y_test);
end

%Calculate the Error
error=error_sum/k_folds;
end
