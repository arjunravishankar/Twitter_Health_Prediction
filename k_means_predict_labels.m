function [pred_labels]=k_means_predict_labels(train_inputs,train_labels,test_inputs)
k=90;
[n_observations,p_features]=size(train_labels);
%Pre allocate predicted label matrix over
pred_labels=zeros(size(test_inputs,1),p_features);

%Loop through each label column to create predicted labels matrix
[idx,C] = kmeans(vertcat(train_inputs,test_inputs),k);
for p=1:p_features
    cluster_labels=zeros(k,1);
    for cluster=1:k
        cluster_labels(cluster)=mean(train_labels(idx(1:size(train_inputs,1))==cluster,p));
    end
    pred_labels(:,p) =cluster_labels(idx((size(train_inputs,1)+1):end));
    
    %Replace NaNs with nearest non Nan cluster_label
    distance=pdist2(test_inputs,C(~isnan(cluster_labels),:),'Euclidean');
    [~,I] = min(distance,[],2);
    NonNan_cluster_labels=cluster_labels(~isnan(cluster_labels));
    backup_test_cluster_labels=NonNan_cluster_labels(I);
    pred_labels(isnan(pred_labels(:,p)),p)=backup_test_cluster_labels(isnan(pred_labels(:,p)));
end

end
