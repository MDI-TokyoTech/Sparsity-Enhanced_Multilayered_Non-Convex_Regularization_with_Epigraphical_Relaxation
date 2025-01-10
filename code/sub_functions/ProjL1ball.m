function[u] = ProjL1ball(u, f, epsilon, varargin)
v = abs(u(:) - f(:));
if sum(v) > epsilon
    sv = sort(v, 'descend');
    cumsv = cumsum(sv);
    J = 1:numel(v);
    thetaCand = (cumsv - epsilon)./J';
    rho = find(((sv - thetaCand)>0), 1, 'last');
    
    if isempty(rho) == 1
        disp('a');
    end
    theta = thetaCand(rho);
    v = v - theta;
    v(v<0) = 0;
    v((u-f)<0) = v((u-f)<0)*-1;
    v = reshape(v, size(u));
    
    u = f + v;
end