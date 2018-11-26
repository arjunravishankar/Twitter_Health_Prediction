% run_FP_Pipeline
% John Bernabei, Arjun Shankar

%% Clear the workspace
clear all

%% Operate pipeline
% Whether to train algorithms? If 1, will train, if 0 will not train.
train=1;
% Whether to test algorithms? If 1, will test, if 0 will not test.
test=1;

% Whether to tune knn to determine optimal k
tune_knn = 0; 
train_RF = 1;

% Can set specific k
set_k = 1;


%% Load data

% Load data, creating variables train_inputs and train_labels
load training_data

% train_inputs is 1019 (counties = n) x 2021 (features = p)
% Column 1 is county code, columns 2-22 are demographics, columns 23-2022
% are LDA topic frequencies from tweets
% train_labels is 1019 (counties = n) x 9 (labels)
% Labels are health outcomes that are necessary to predict

%% Set up overall model parameters
% Get dimensions of dataset
[n, p] = size(train_inputs);
labels = size(train_labels,2);

% Set number of folds for CV
n_folds = 10;

%% Process data

train_inputs = (train_inputs-mean(train_inputs))./max((train_inputs-mean(train_inputs)));

%% Segment into separate folds for CV
partitions = make_xval_partition(n, n_folds);

%% Dimensionality reduction?
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(train_inputs(:,23:2021));
X_red = SCORE(:,1:20);
%X_reduce = train_inputs(:,2:500);
X_reduce = [X_red,train_inputs(:,2:22)];
%% Train algorithms
if train==1
    % Below we will select appropriate algorithms to ensemble in order to make
    % the optimal predictions

    % Each method has a boolean which can be selected to determine how to
    % handle the model, switch it on or off, or tune it's parameters

    % Adjust generative method params


    % Adjust discriminative method params
    %

    % Adjust instance-based method params
    % K nearest neighbors
    %% Generative method (Naive Bayes, k means, gaussian mixtures)
    
    
    %% Discriminative method (regression, SVM, random forest?)
    
    % Below we will implement random forest
    % Define number of trees
    num_trees = 40;
    % Train random forest to produce model
    if train_RF==1
        %for i=1:n_folds
            for j=1:labels
                X_train_red = X_reduce(partitions ~=1,:);
                Y_train = train_labels(partitions ~=1,:);
                X_train = train_inputs(partitions ~=1,:);
                mdl(j).data = TreeBagger(num_trees,X_train_red,Y_train(:,j),...
                            'oobpred','On','Method','regression',...
                            'OOBVarImp','on');
                        
                figure(j);clf
                plot(oobError(mdl(j).data))
                xlabel('number of grown trees')
                ylabel('out-of-bag classification error')
            end
        %end
    end
    %% Instance-based method (knn, kernel regression)
    % Below we implement k nearest neighbors
    % Select whether to tune models or not
    if tune_knn == 1
        % Will select optimal k from [1 2 5 10 20 50 100]
        k = 1; % change dis
    else
        k = set_k;
    end
    
    % Somethind with novelty?
    % Below we will implement deep net
    
end
%% Ensemble methods to make predictions
if test==1
    for j=1:labels
       Y_train = train_labels(partitions ~=1,:);
       X_test_reduce = X_reduce(partitions ==1,:);
       X_test = train_inputs(partitions ==1,:);
       Y_pred_knn(:,j) = k_nearest_neighbours(X_train,Y_train(:,j),X_test,7,'l2');
       Y_pred_rf(:,j) = predict(mdl(j).data,X_test_reduce);
    end

    %% Test based on remaining data
    %Y_pred_rf=cellfun(@str2num,Y_pred_svm)
    Y_pred = (Y_pred_knn+Y_pred_rf)./2;
    Y_test = train_labels(partitions ==1,:);
    error = error_metric(Y_pred,Y_test)

end
%% Calculate error metrics
% This is the error metric given
%error=error_metric(predicted_labels,true_labels);

%% Plot