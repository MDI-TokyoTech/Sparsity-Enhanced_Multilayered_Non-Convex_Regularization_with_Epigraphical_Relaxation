% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function y = ProxNN(x, gamma , weight)

[U, S, V] = svd(x, 0);

if ~exist('weight','var')
    weight = ones(1,size(S,1));
end

% y = U*max(S-gamma,0)*V';
Sthre = diag(max(0, diag(S) - gamma*weight'));

y = U*Sthre*V';
