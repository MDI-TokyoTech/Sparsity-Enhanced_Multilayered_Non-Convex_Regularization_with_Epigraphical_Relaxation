function[u] = Proj_NormBall(u, f, ballSort, epsilon, varargin)

if strcmp(ballSort, 'L2ball')
    radius = norm(u(:) - f(:),2);
    if radius > epsilon
        u = f + (epsilon/radius)*(u - f);
    end
elseif strcmp(ballSort, 'L1ball')
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
elseif strcmp(ballSort, 'WL1ball')
    worg = varargin{1}(:); % weight
    vorg = (u(:) - f(:));
    v = abs(vorg);
    if sum(v.*worg) > epsilon
        vw = v./worg;
        [vw, Ind] = sort(vw, 'descend');
        v = v(Ind);
        L = numel(u);
        w = worg(Ind);
        gamma = zeros(L+1,1);
        gamma(1) = L;
        l = 1;
        while l <= L
            lambdaStar = l;
            jstar = find(vw(1:gamma(l)) > (sum(v(1:gamma(l)).*w(1:gamma(l))) - epsilon)/sum(w(1:gamma(l)).^2), 1, 'last');
            if jstar == gamma(l)
                break;
            else
                %             gamma(l+1)
                %             jstar
                gamma(l+1) = jstar;
            end
            l = l + 1;
        end
        %lambdaStar= l - 1;
        gamma = gamma(lambdaStar);
        phat = v(1:gamma) - (sum(v(1:gamma).*w(1:gamma)) - epsilon)/sum(w(1:gamma).^2)*w(1:gamma);
        result = zeros(L,1);
        result(1:gamma) = phat;
        result(Ind) = result;
        v = sign(vorg).*result;
        u = f + reshape(v, size(u));
    end
elseif strcmp(ballSort, 'NNball')
    [ U, D, V ] = svd(u,0);
    Dtld = diag( ProjNormBall(diag(D), zeros(size(diag(D))), 'L1ball', epsilon, varargin) );
    u = U*Dtld*V';
end