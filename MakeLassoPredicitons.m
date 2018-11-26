y_lasso_predictions=zeros(n,numberoflabels);

for t=1:size(train_labels,2)
    B=B_Cell{t};
    FitInfo=FitInfoCell{t};
    coef = B(:,20);
    coef0 = FitInfo.Intercept(20);
    y_lasso_predictions(:,t) = train_inputs*coef + coef0;
end
