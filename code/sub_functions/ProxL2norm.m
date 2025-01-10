% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)
% 
% f(x) = (É /2)|| x - v ||_2^2 
% prox_{É¡f}(x) = argmin_y É¡(É /2)|| y - v ||_2^2 + (1/2) || x - y ||_2^2
%              = argmin_y É¡(É /2)|| y - v ||_2^2 + (1/2) || y - x ||_2^2

% (åˆéÆ) Å›/Å›y ( || Ax - b ||_2^2 ) = A^T 2(Ax-b)

% Å›/Å›y ( É¡(É /2)|| y - v ||_2^2 + (1/2) || y - x ||_2^2 )
% = É¡É ( y - v ) + ( y - x ) = ( 1 + É¡É  )y - ( É¡É v + x ) = 0
% ( 1 + É¡É  )y = É¡É v + x
% y = (É¡É v + x)/( 1 + É¡É  )
%

function y = ProxL2norm(x, gamma, v, mu)

if ~exist('mu','var')
    mu = 1;
end

if ~exist('v','var')
    v = zeros(size(x));
end

y = (x + mu*gamma*v)/( 1 + mu*gamma );