function y = Prox_L1norm(x,T)

% if ~exist('v','var')
%     v = zeros(size(x));
% end

% y = v + sign(x-v).*max(abs(x-v)-T,0);
y = sign(x).*max(abs(x)-T,0);
