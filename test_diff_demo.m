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
load features

final_feats = []
for j = 1:9
   final_feats = [final_feats, mMSEpred(j).data] 
end

fin_feats = unique(final_feats);
final_feats = fin_feats(1<histc(final_feats,fin_feats))
%load tweet_ind

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
%X_tweet = train_inputs(:,tweet_ind);
X_tweet = train_inputs(:,22:end);


% Extract non-redundant demographics
% Demographic columns 7-10, 21 data are contained in column 11
X_demo = train_inputs(:,[1:6,11:20]);

%% Reduce data
% Tweet data then PCA
%tweet_data_red = X_tweet;
tweet_var = var(X_tweet);
tweet_ind = find(tweet_var>=(mean(tweet_var)));
tweet_data_red = X_tweet(:,tweet_ind);
% PCA tweet data
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(X_tweet(:,(final_feats(final_feats>21)-21)));
X_tweet_pca = SCORE(:,1:30);
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(X_demo);
X_demo_pca = SCORE(:,1:15);

%% Partition data 
% Get partitions
partitions = make_xval_partition(n, n_folds);

% Make X train and test
X_train_raw = train_inputs(partitions ~=1,:);
X_tweet_train = tweet_data_red(partitions ~=1,:);
X_tweet_pca_train = X_tweet_pca(partitions ~=1,:);
X_demo_train = X_demo(partitions ~=1,:);
X_demo_pca_train = X_demo_pca(partitions ~=1,:);

X_comp_train = [X_demo_train,X_tweet_pca_train];
Z_comp_train = zscore(X_comp_train);

X_test_raw = train_inputs(partitions ==1,:);
X_tweet_test = tweet_data_red(partitions ==1,:);
X_tweet_pca_test = X_tweet_pca(partitions ==1,:);
X_demo_test = X_demo(partitions ==1,:);
X_demo_pca_test = X_demo_pca(partitions ==1,:);

X_comp_test = [X_demo_test, X_tweet_pca_test];
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

%% Do lasso

for j=1:9
    [B,FitInfo] = lasso(X_train_raw(:,final_feats),Y_train(:,j),'Alpha',0.5,'MaxIter',1e4,'CV',5)%,'PredictorNames',string([1:size(X_comp_train,2)]));
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    coef0 = FitInfo.Intercept(idxLambda1SE);
    Y_est_el(:,j) = X_train_raw(:,final_feats)*coef + coef0;
    Y_pred_el(:,j) = X_test_raw(:,final_feats)*coef + coef0;
    %idxLambdaMinMSE = FitInfo.IndexMinMSE;
    %mMSEpred(j).data = cellfun(@str2num,FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0))
   if j ~= 3 && j ~= 8
        Y_est_el(:,j) = round(Y_est_el(:,j));
        Y_pred_el(:,j) = round(Y_pred_el(:,j));
   else
       Y_est_el(:,j) = round(Y_est_el(:,j),1);
        Y_pred_el(:,j) = round(Y_pred_el(:,j),1); 
   end
end

%% neural net

for j = 1:9
    net = feedforwardnet(30);
    net.layers{1}.transferFcn = 'poslin';% 1st layer activation function is logistic sigmoid
    net.performFcn = 'crossentropy' ;  %specify loss function appropriate for classification
    net.performParam.regularization = 10e-3 ;  % regularization parameter
    net = trainlm(net,Z_comp_train',Y_train(:,j)'); % train with backpropagation
    Y_est_ne(:,j) = net(Z_comp_train');
    Y_pred_ne(:,j)= net(Z_comp_test');
    j
end


%% Discriminative model - random forest

for j = 1:labels
    mdl_rf(j).data = TreeBagger(70,X_comp_train,Y_train(:,j),...
                'oobpred','On','Method','regression',...
                'OOBVarImp','on');   
    Y_est_rf(:,j) = predict(mdl_rf(j).data,X_comp_train);
    Y_pred_rf(:,j) = predict(mdl_rf(j).data,X_comp_test);
   if j ~= 3 && j ~= 8
       Y_est_rf(:,j) = round(Y_est_rf(:,j));
        Y_pred_rf(:,j) = round(Y_pred_rf(:,j));
   else
       Y_est_rf(:,j) = round(Y_est_rf(:,j),1);
        Y_pred_rf(:,j) = round(Y_pred_rf(:,j),1); 
   end
end


%% Instance based model - knn

for j = 1:labels
    Y_est_kr(:,j) = k_nearest_neighbours(Z_comp_train,Y_train(:,j),Z_comp_train,20,'l2');
    Y_pred_kr(:,j) = k_nearest_neighbours(Z_comp_train,Y_train(:,j),Z_comp_test,20,'l2');
end

%% Ensemble

for j=1:labels
   mdl_ens(j).data = TreeBagger(30,[Y_est_rf, Y_est_kr, Y_est_ne, Y_est_el],Y_train(:,j),...
                'oobpred','On','Method','regression',...
                'OOBVarImp','on'); 
   Y_pred_ens(:,j) = predict(mdl_ens(j).data,[Y_pred_rf, Y_pred_kr, Y_pred_ne, Y_pred_el]);
end

%% Return error
error_ens = error_metric(Y_pred_ens,Y_test)
error_rf  = error_metric(Y_pred_rf,Y_test)
error_kr  = error_metric(Y_pred_kr,Y_test)
error_el  = error_metric(Y_pred_el,Y_test)
error_ne  = error_metric(Y_pred_ne,Y_test)
