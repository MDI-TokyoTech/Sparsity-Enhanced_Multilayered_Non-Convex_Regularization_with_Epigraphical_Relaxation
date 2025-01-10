function y = ProxRank(x, K)

[U, S, V] = svd(x, 0);
%disp(rank(S))
% x = U*S*V'

% y = U*max(S-gamma,0)*V';
s = diag(S);
s(K+1:end) = 0;
Stld = diag(s);

y = U*Stld*V';
