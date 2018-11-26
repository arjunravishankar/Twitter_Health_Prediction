function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
% Redefine variables
X_train = train_inputs;
Y_train = train_labels;
X_test  = test_inputs;

[n, p] = size(X_train);
labels = size(Y_train,2);

X_train_reduce = (X_train(:,2:end));
X_test_reduce  = (X_test(:,2:end));

num_trees = 40;

for j=1:labels
                mdl(j).data = TreeBagger(num_trees,X_train_reduce,Y_train(:,j),...
                            'oobpred','On','Method','regression',...
                            'OOBVarImp','on');
                Y_pred_rf(:,j) = predict(mdl(j).data,X_test_reduce);
end

pred_labels = Y_pred_rf

%pred_labels=randn(size(test_inputs,1),size(train_labels,2));

end

