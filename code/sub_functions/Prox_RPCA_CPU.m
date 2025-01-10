function [znew] = Prox_RPCA_CPU(z, gamma, lambda1, lambda2)
%PROX_RPCA_CPU prox(lambda1*||L||_* + lambda2*||S||_1)

rows = size(z, 1);
l = z(1:rows/2, :);
s = z(1+rows/2:end, :);

M = sqrt(size(l, 1)/ 12)*4;
N = sqrt(size(l, 1)/ 12)*3;
prox_mat =  ProxNN(reshape(l, [M, N]), gamma, lambda1);

znew = zeros(size(z));
znew(1:rows/2, :) = prox_mat(:);
znew(1+rows/2:end, :) = Prox_L1norm(s, gamma*lambda2);
end

