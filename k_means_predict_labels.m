function [pred_labels]=k_means_predict_labels(train_inputs,train_labels,test_inputs)

[n_observations,p_features]=size(train_labels);
%Pre allocate predicted label matrix over
pred_labels=zeros(size(test_inputs,1),p_features);

%Loop through each label column to create predicted labels matrix
for p=1:p_features
    
    [idx,C] = kmeans(train_inputs,length(unique(train_labels(:,p))));
    cluster_labels=zeros(length(unique(train_labels(:,p))),1);
    
    for cluster=1:length(unique(train_labels(:,p)))
        cluster_labels(cluster)=mean(train_labels(idx==cluster,p));
    end
    
    distance=pdist2(test_inputs,C,'Euclidean');
    [~,I] = min(distance,[],2);
    pred_labels(:,p) = cluster_labels(I);
end

end