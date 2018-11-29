clear all
load training_data

[n, p] = size(train_inputs);
labels = size(train_labels,2);



X_train = train_inputs;
Y_train = train_labels;

for j=1:9
    [B,FitInfo] = lasso(X_train,Y_train(:,j),'Alpha',1,'MaxIter',1e5,'CV',5,'PredictorNames',string([1:size(X_train,2)]));
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    coef0 = FitInfo.Intercept(idxLambda1SE);
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    mMSEpred(j).data = cellfun(@str2num,FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0))
end

save('features.mat','mMSEpred')