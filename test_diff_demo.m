% run_FP_Pipeline
% John Bernabei, Arjun Shankar

%% Clear the workspace
clear all

%% Operate pipeline
% Whether to train algorithms? If 1, will train, if 0 will not train.
train=1;
% Whether to test algorithms? If 1, will test, if 0 will not test.
test=1;

%% Load data

% Load data, creating variables train_inputs and train_labels
load training_data
load tweet_ind

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
% Extract tweets
X_tweet = train_inputs(:,tweet_ind);
X_all_tweet = train_inputs(:,22:end);


% Extract non-redundant demographics
% Demographic columns 7-10, 21 data are contained in column 11
X_demo = train_inputs(:,[1:6,11:20]);

%% Reduce data
% Tweet data then PCA
tweet_data_red = X_tweet;
% PCA tweet data
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(X_all_tweet);
X_tweet_pca = SCORE(:,1:25);
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(X_demo);
X_demo_pca = SCORE(:,1:15);

%% Partition data 
% Get partitions
partitions = make_xval_partition(n, n_folds);

% Make X train and test
X_tweet_train = tweet_data_red(partitions ~=1,:);
X_tweet_pca_train = X_tweet_pca(partitions ~=1,:);
X_demo_train = X_demo(partitions ~=1,:);
X_demo_pca_train = X_demo_pca(partitions ~=1,:);

X_comp_train = [X_tweet_pca_train, X_demo_train];
Z_comp_train = zscore(X_comp_train);

X_tweet_test = tweet_data_red(partitions ==1,:);
X_tweet_pca_test = X_tweet_pca(partitions ==1,:);
X_demo_test = X_demo(partitions ==1,:);
X_demo_pca_test = X_demo_pca(partitions ==1,:);

X_comp_test = [X_tweet_pca_test, X_demo_test];
Z_comp_test = zscore(X_comp_test);

% Z score appropriate sub-data
Z_demo_train = zscore(X_demo_train);
Z_demo_test = zscore(X_demo_test);

Z_tweet_train = zscore(X_tweet_train);
Z_tweet_test = zscore(X_tweet_test);


% Make Y train and test
Y_train = train_labels(partitions ~=1,:);
Y_test = train_labels(partitions ==1,:);

%% Generative model - kmeans
% Modeled on composite

kmeans_total = kmeans([Z_demo_train;Z_demo_test],3);
Y_est_km  = kmeans_total(partitions ~=1);
Y_pred_km = kmeans_total(partitions ==1);

%% Discriminative model - random forest

for j = 1:labels
    mdl_rf(j).data = TreeBagger(160,X_comp_train,Y_train(:,j),...
                'oobpred','On','Method','regression',...
                'OOBVarImp','on');   
    Y_est_rf(:,j) = predict(mdl_rf(j).data,X_comp_train);
    Y_pred_rf(:,j) = predict(mdl_rf(j).data,X_comp_test);
end

%% Do lasso

for j=1:9
    [B,FitInfo] = lasso(X_comp_train,Y_train(:,j),'Alpha',0.1,'MaxIter',1e3,'CV',5);
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    coef0 = FitInfo.Intercept(idxLambda1SE);
    Y_est_el(:,j) = X_comp_train*coef + coef0;
    Y_pred_el(:,j) = X_comp_test*coef + coef0;
    j
end


%% Instance based model - knn

for j = 1:labels
    Y_est_kr(:,j) = k_nearest_neighbours(Z_comp_train,Y_train(:,j),Z_comp_train,20,'l2');
    Y_pred_kr(:,j) = k_nearest_neighbours(Z_comp_train,Y_train(:,j),Z_comp_test,20,'l2');
end

%% Ensemble

for j=1:labels
   mdl_ens(j).data = TreeBagger(25,[Y_est_rf, Y_est_kr, Y_est_el],Y_train(:,j),...
                'oobpred','On','Method','regression',...
                'OOBVarImp','on'); 
   Y_pred_ens(:,j) = predict(mdl_ens(j).data,[Y_pred_rf, Y_pred_kr, Y_pred_el]);
end

%% Return error
error_ens = error_metric(Y_pred_ens,Y_test)
error_rf  = error_metric(Y_pred_rf,Y_test)
error_kr  = error_metric(Y_pred_kr,Y_test)
error_el  = error_metric(Y_pred_el,Y_test)
