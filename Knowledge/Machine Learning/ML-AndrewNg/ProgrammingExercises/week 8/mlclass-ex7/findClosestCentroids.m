function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


% for i = 1:length(X),
% for j = 1:K,
% T(j) = (X(i,1)-centroids(j,1))^2 + (X(i,2)-centroids(j,2))^2;
% end;
% [v,index] = min(T);
% idx(i) = index
% end;

X_k = idx;
for i = 1:K,  
    X_item = X - centroids(i, :);  
    X_item = X_item.*X_item;  
    X_item = sum(X_item, 2);  
    if i == 1,  
        X_k = X_item;  
    else,  
        X_k = [X_k X_item];
    end;  
end;      
  
[val, ind] = min(X_k, [], 2);   
idx = ind; 





% =============================================================

end

