% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function y = ProxNN2(x, gamma)
for i = 1:2
[U, S, V] = svd(x(:,:,i), 0);

if ~exist('weight','var')
    weight = 1.5*ones(1,size(S,1));
end

% y = U*max(S-gamma,0)*V';
Sthre = diag(max(0, diag(S) - gamma*weight'));

y(:,:,i) = U*Sthre*V';
end

