function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = 1 ./(1 + exp(-X * theta));
J = -sum(y .* log(h) + (1 - y) .* log(1 - h)) / m + lambda /(2 * m) * sum(theta([2:size(theta,1)]) .^ 2);

% regularize the parameter
j = length(theta);

for iter = 1: j,
if iter == 1,
grad_(iter) = sum((h - y) .* X(:,iter))/ m;
else,
grad_(iter) = sum((h - y) .* X(:,iter))/ m + lambda/ m * theta(iter);
end;
end;

for iter = 1: j,
grad(iter) = grad_(iter);
end




% =============================================================

end
