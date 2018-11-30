%% test neural net 

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
   final_feats = [final_feats, mMSEpred(j).data]; 
end

fin_feats = unique(final_feats);
final_feats = fin_feats(1<histc(final_feats,fin_feats));
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
X_tweet_pca = SCORE(:,1:10);
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

Z_tot_comp = zscore([X_demo, X_tweet_pca]);

Z_comp_train2 = Z_tot_comp(partitions ~=1,:);
Z_comp_test2 = Z_tot_comp(partitions ==1,:);


% Z score appropriate sub-data
Z_demo_train = zscore(X_demo_train);
Z_demo_test = zscore(X_demo_test);

Z_tweet_train = zscore(X_tweet_train);
Z_tweet_test = zscore(X_tweet_test);


% Make Y train and test
Y_train = train_labels(partitions ~=1,:);
Y_test = train_labels(partitions ==1,:);

%% neural net

for j = 1:9
    net = feedforwardnet(13);
    net.layers{1}.transferFcn = 'poslin';% 1st layer activation function is logistic sigmoid
    net.performFcn = 'crossentropy' ;  %specify loss function appropriate for classification
    net.performParam.regularization = 10e-2 ;  % regularization parameter
    net = trainlm(net,Z_comp_train2',Y_train(:,j)'); % train with backpropagation
    Y_est_ne(:,j) = net(Z_comp_train2');
    Y_pred_ne(:,j)= net(Z_comp_test2');
    j
end

%% Return error
error_ne  = error_metric(Y_pred_ne,Y_test)
