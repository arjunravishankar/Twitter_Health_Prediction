
% Calculate Mapped Training Data and Mapping to convert other feature
% matrices to best LDA components in feature space
for j=1:9
[X_lda_train{j}, Mapping{j}]= lda(train_inputs,train_labels(:,j), 400);
end

% Convert any test feature matrix to best LDA components 

% Make sure data is zero mean
feature_mean=mean(test_inputs, 1);
X = bsxfun(@minus, test_inputs, feature_mean);

% Compute mapped data
for j=1:3
X_lda_test{j} = X * Mapping{j}.M;
end