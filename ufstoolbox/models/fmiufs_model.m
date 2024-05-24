function [selected_features] = fmiufs_model(X)
%FMIUFS_MODEL Summary of this function goes here
%   Detailed explanation goes here
X = X - min(X);
X = X ./ max(X);
selected_features = ufs_FMI(X,1.0);
end

