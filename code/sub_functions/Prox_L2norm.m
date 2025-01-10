function[ y ] = Prox_L2norm(x, gamma , v,  mu)

if ~exist('v','var')
    v = zeros(size(x));
end

if ~exist('mu','var')
    mu = 1;
end

y = (gamma*mu*v + x)/(mu*gamma+1);