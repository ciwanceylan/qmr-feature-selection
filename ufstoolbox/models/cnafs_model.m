function [I] = cnafs_model(X)
%DGUFS Summary of this function goes here
%   Detailed explanation goes here
[~, ~, I, ~] = CNAFS(X.', 5, 1.0, 100, 100, 100, 1.0, 500, 20);
end