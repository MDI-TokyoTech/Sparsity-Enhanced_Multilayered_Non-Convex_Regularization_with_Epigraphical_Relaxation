function [x_star, s_star] = Proj_Epigraph_L1(x, s)
% Epigraph_L1 is the set of s when ||x||_1 <= s
% input1 : vector x in R^{N}
% input2 : scalar s in R
% 
% If s is in the epigraph
%   [x_star, s_star] = [x, s]
%
% If s is not in the epigraph
%   x_star = soft_thresholding(x, lambda_star)
%   s_star = lambda_star + s
%
% where lambda_star is a solution of the following function:
%   f(x) = ||soft_thresholding(x, lambda)||_1 - lambda - s
% for any lambda > 0
% See Appendix.E in the paper [Epigraphical_Relaxation_for_Minimizing_Layered_Mixed_Norms, Kyochi, 2021]

absx = abs(x);
L1norm = sum(absx);

if L1norm <= s
    % s is in the epigraph
    x_star = x;
    s_star = s;
else
    % s is not in the epigraph
    absx = sort(absx, 'descend');
    if s < -absx(1)
        lambda_star = -s;
    else
        % expand by 0
        N = length(absx);
        absx(N+1) = 0;
        % absxex = zeros(N+1,1);
        % absxex(1:N) = absx;

        sum_absx = 0;
        for n = 1:N
            sum_absx = sum_absx + absx(n); % sum_absx = sum(absxex(1:n));
            booll = ( sum_absx - (n+1)*absx(n) ) <= s;
            boolr =  s < ( sum_absx - (n+1)*absx(n+1) );
            if booll && boolr
                lambda_star = (sum_absx - s)/(n+1);
                break;
            end
        end
    end
    
    x_star = Prox_L1norm(x, lambda_star);
    s_star = s + lambda_star;

    % % 更新されたs_starがx_starのL1エピグラフの境界に張り付いているかチェック
    % if norm(sum(abs(x_star)) - s_star) > 1e-10
    %     msg = 'Error occurred.';
    %     error(msg)
    % end
end