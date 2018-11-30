function [pred_labels]=predict_labels(train_inputs,train_labels,test_inputs)
%% Load and import data
% Get given X train/test, Y_train
X_train = train_inputs;
X_test  = test_inputs;

Y_train = train_labels;

% Get other training data
load train_data
load features

X_train_tot = [X_train];
Y_train_tot = [Y_train];

X_tot = [X_train;X_test];

% Get sizes of data
[n_train, p] = size(X_train_tot);
[n_test, p] = size(X_test);
labels = size(train_labels,2);

%% get final features
final_feats = [];
for j = 1:9
   final_feats = [final_feats, mMSEpred(j).data];
end

fin_feats = unique(final_feats);
final_feats = fin_feats(1<histc(final_feats,fin_feats));

%% Process data
% Extract tweets
%X_tweet = train_inputs(:,tweet_ind);
X_tweet = X_tot(:,22:end);


% Extract non-redundant demographics
% Demographic columns 7-10, 21 data are contained in column 11
X_demo = X_tot(:,[1:6,11:20]);

%% Reduce data
% PCA tweet data
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(X_tweet(:,(final_feats(final_feats>21)-21)));
X_tweet_pca = SCORE(:,1:30);

%% Get final data
X_demo_train = X_demo((1:n_train),:);
X_tweet_pca_train = X_tweet_pca((1:n_train),:);
X_comp_train = [X_demo_train,X_tweet_pca_train];
Z_comp_train = zscore(X_comp_train);

X_demo_test = X_demo((n_train+1):end,:);
X_tweet_pca_test = X_tweet_pca((n_train+1):end,:);
X_comp_test = [X_demo_test, X_tweet_pca_test];
Z_comp_test = zscore(X_comp_test);

Z_tot_comp = zscore([X_demo, X_tweet_pca]);
Z_comp_train2 = Z_tot_comp((1:n_train),:);
Z_comp_test2 = Z_tot_comp((n_train+1):end,:);


Y_train = Y_train_tot;

%% Do lasso
for j=1:9
    [B,FitInfo] = lasso(X_train_tot(:,final_feats),Y_train(:,j),'Alpha',0.5,'MaxIter',1e4,'CV',5)%,'PredictorNames',string([1:size(X_comp_train,2)]));
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    coef0 = FitInfo.Intercept(idxLambda1SE);
    Y_est_el(:,j) = X_train_tot(:,final_feats)*coef + coef0;
    Y_pred_el(:,j) = X_test(:,final_feats)*coef + coef0;
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
    net = feedforwardnet(14);
    net.layers{1}.transferFcn = 'poslin';% 1st layer activation function is logistic sigmoid
    net.performFcn = 'crossentropy' ;  %specify loss function appropriate for classification
    net.performParam.regularization = 10e-2 ;  % regularization parameter
    net = trainlm(net,Z_comp_train2',Y_train(:,j)'); % train with backpropagation
    Y_est_ne(:,j) = net(Z_comp_train2');
    Y_pred_ne(:,j)= net(Z_comp_test2');
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
    Y_est_kr(:,j) = k_nearest_neighbours(Z_comp_train2,Y_train(:,j),Z_comp_train2,20,'l2');
    Y_pred_kr(:,j) = k_nearest_neighbours(Z_comp_train2,Y_train(:,j),Z_comp_test2,20,'l2');
end

%% Ensemble

for j=1:labels
   mdl_ens(j).data = TreeBagger(30,[Y_est_rf, Y_est_kr, Y_est_ne, Y_est_el],Y_train(:,j),...
                'oobpred','On','Method','regression',...
                'OOBVarImp','on'); 
   Y_est_ens(:,j) = predict(mdl_ens(j).data,[Y_est_rf, Y_est_kr, Y_est_ne, Y_est_el])
   Y_pred_ens(:,j) = predict(mdl_ens(j).data,[Y_pred_rf, Y_pred_kr, Y_pred_ne, Y_pred_el]);
end

%% Return labels
error_ens = error_metric(Y_est_ens,Y_train)
error_rf  = error_metric(Y_est_rf,Y_train)
error_kr  = error_metric(Y_est_kr,Y_train)
error_el  = error_metric(Y_est_el,Y_train)
error_ne  = error_metric(Y_est_ne,Y_train)


pred_labels = Y_pred_ens;

end

