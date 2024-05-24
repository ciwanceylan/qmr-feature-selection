function [idx] = ls_model(X)
%LS_MODEL Summary of this function goes here
%   Detailed explanation goes here
W = constructW(X, []);
scores = LaplacianScore(X, W);

[out,idx] = sort(scores, 'descend');
end

