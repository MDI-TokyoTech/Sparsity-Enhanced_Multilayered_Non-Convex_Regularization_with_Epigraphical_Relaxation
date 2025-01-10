function [znew] = Prox_RPCA_color_CPU(z, gamma, lambda1, lambda2)
%PROX_RPCA_CPU prox(||L||_* + ||S||_1)

rows = size(z, 1);
n = sqrt(rows/2);
prox_mat =  ProxNNcolor(reshape(z(1:rows/2, :), [n, n]), gamma, lambda1);

znew = zeros(size(z));
znew(1:rows/2, :) = prox_mat(:);
znew(1+rows/2:end, :) = Prox_L1norm(z(1+rows/2:end, :), gamma*lambda2);
end

