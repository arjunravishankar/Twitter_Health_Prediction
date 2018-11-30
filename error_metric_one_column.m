%this function calculates average of normalized RMSE of all labels
%first input is predicted labels, second input is true labels corresponding
%to your predictions

function error=error_metric_one_column(predicted_labels,true_labels,label_number)
    scale=[571.0000   29.0000    3.3000   12.0000   17.0000   33.0000   24.0000    3.8000   30.0000]; %this is MaxMin of training labels
    error=sum(sqrt(sum((predicted_labels-true_labels).^2)/size(true_labels,1))./scale(label_number));
end