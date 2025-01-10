function [l, s, time_opt, convergeNum] = RPCA_PDS(u_obsv, para, u_org, rangeConstraint)
%==================================
%  input: 2D shifted signal
% output: Low-rank signal, Sparse noise
%==================================
% u = [l, s]^T
% A = [I I]
% u* = argmin_u 1/2|| A(u) - y ||_F^2 + lambda_1*|| l ||_*, s.t. || s ||_1 <= lambda_2
%
% f(u) = 0
% g(u) = 0
% h(Lu) = 1/2|| A(u) - u_obsv ||_F^2 + lambda_1*|| l ||_* + iota_{||.||_1 <= lambda_2}(s)
% L = [I I]
%     [I O]
%     [O I]
%
% prox_g (x) = argmin_y g(y) + (1/2)|| x - y ||_2^2 
%            = argmin_y (1/2)|| x - y ||_2^2
%            = x
%
% if rangeConstraint ON
% h(Lu) = 1/2|| A(u) - u_obsv ||_F^2 + lambda_1*|| l ||_* + iota_{||.||_1 <= lambda_2}(s) + iota_{[a,b]}(l)
% L = [I I]
%     [I O]
%     [O I]
%     [I O]

%% Settings
%-----------------------------------------------------
% step size
%-----------------------------------------------------
l1 = para.lambda_1;
l2 = para.lambda_2;
theta = para.theta_1;

gamma1 = 0.01; % parameter of PDS
gamma2 = 0.99/(3*gamma1); % parameter of PDS (||L||op^2 = eigs([2*I, I; I, 2*I], 1) = 3)

%-----------------------------------------------------
% linear operators, proximity operators
%-----------------------------------------------------
[rows, cols, dim] = size(u_obsv);
N = rows*cols*dim;

% df/du
df = @(z) {0, 0};
% proximity operator of function g
prox_f1 = cell(1,2);
prox_f1{1} = @(z, gamma) z;
prox_f1{2} = @(z, gamma) z;
% proximity operator of function h
if ~exist('rangeConstraint', 'var')
    prox_f2 = cell(1,3);
    prox_f2{1} = @(z, gamma) ProxL2norm(z, gamma, u_obsv);
    prox_f2{2} = @(z, gamma) ProxNN(z, gamma, l1*para.mu);
    prox_f2{3} = @(z, gamma) ProjL1ball(z, zeros(size(z)), l2*para.mu);
    if dim > 1
        prox_f2{2} = @(z, gamma) ProxNNcolor(z, gamma*l1*para.mu); % for color image
    end
    
    L = @(z) { z{1} + z{2} , z{1} , z{2} };
    Lt = @(z) { z{1} + z{2} , z{1} + z{3} };
else
    prox_f2 = cell(1,4);
    prox_f2{1} = @(z, gamma) ProxL2norm(z, gamma, u_obsv);
    prox_f2{2} = @(z, gamma) ProxNN(z, gamma, l1*para.mu);
    prox_f2{3} = @(z, gamma) ProjL1ball(z, zeros(size(z)), l2*para.mu);
    prox_f2{4} = @(z, gamma) Proj_RangeConstraint(z, rangeConstraint);
    if dim > 1
        prox_f2{2} = @(z, gamma) ProxNNcolor(z, gamma*l1*para.mu); % for color image
    end
    
    L = @(z) { z{1} + z{2} , z{1} , z{2} , z{1}  };
    Lt = @(z) { z{1} + z{2} + z{4}, z{1} + z{3} };
end


%-----------------------------------------------------
% Initial values
%-----------------------------------------------------
x = cell(1, 2);
x{1} = u_obsv; % l
x{2} = zeros(size(u_obsv)); % s

y = L(x);
xnum = numel(x);
ynum = numel(y);

%% main
%-----------------------------------------------------
% main loop
%-----------------------------------------------------
disp(mfilename);

converge_rate = zeros(1, para.maxiter);
RMSE_L = zeros(1, para.maxiter);
progressWindow = figure("Name", mfilename);
progressWindowRMSE = figure("Name", strcat(mfilename, " - RMSE"));
ax = axes("Parent", progressWindow);
ax2 = axes("Parent", progressWindowRMSE);
movegui(progressWindow, 'center');
movegui(progressWindowRMSE, 'east');

tic;
for i = 1:para.maxiter
    % primal update
    xpre = x;
    x = cellfun(@(z1, z2, z3) z1 - gamma1*z2 - gamma1*z3, x, df(x), Lt(y), 'UniformOutput', false);
    for j = 1:xnum
        x{j} = prox_f1{j}(x{j}, gamma1);
    end
    
    % dual update
    Ltemp = L(cellfun(@(z1,z2) 2*z1 - z2, x, xpre, 'UniformOutput', false));
    for j = 1:ynum
        Ltemp{j} = y{j} + gamma2 * Ltemp{j};
        y{j} = Ltemp{j} - gamma2 * prox_f2{j}(Ltemp{j}/gamma2, 1/gamma2);
    end

    % check convergence
    converge_rate(i) = norm(reshape(x{1} - xpre{1}, [N, 1])) + ...
                         norm(reshape(x{2} - xpre{2}, [N, 1]));
    fprintf('iter: %d, Error(X) = %f\n', i, converge_rate(i));
    RMSE_L(i) = sqrt( sum(sum(sum((x{1} - u_org).^2))) / N );

    % view progress
    if mod(i, 100) == 0
        semilogy(ax, 1:i, converge_rate(1:i));
        ylabel(ax, '$\|(X_{n+1} - X_n\|$','Interpreter','latex')
        xlabel(ax, 'iteration')
        title(ax,'converge rate')

        semilogy(ax2, 1:i, RMSE_L(1:i));
        ylabel(ax2, 'RMSE','Interpreter','latex')
        xlabel(ax2, 'iteration')
        title(ax2,'converge rate')
        drawnow limitrate
    end

    % exit loop
    if converge_rate(i) < para.stopcri
        semilogy(ax, 1:i, converge_rate(1:i));
        ylabel(ax, '$\|(X_{n+1} - X_n\|$','Interpreter','latex')
        xlabel(ax, 'iteration')
        title(ax,'converge rate')

        semilogy(ax2, 1:i, RMSE_L(1:i));
        ylabel(ax2, 'RMSE','Interpreter','latex')
        xlabel(ax2, 'iteration')
        title(ax2,'converge rate')
        drawnow
        break;
    end
end
time_opt = toc;
convergeNum = i;
l = x{1};
s = x{2};

save(sprintf('%s/mat/%s/%02d/%s/theta%.3f_lambda1_%.3f_lambda2_%.3f.mat', para.currentDir, para.imageName,...
    para.exNumber, para.methodName, theta, l1, l2), ...
    "x", "l", "s", "converge_rate", "convergeNum", "time_opt", "RMSE_L", "para");
%-----------------------------------------------------
% Save progress
%-----------------------------------------------------
graphImageName = sprintf('%s/images/%s/%02d/%s/progressWindow_theta%.3f_lambda1_%.3f_lambda2_%.3f.png', ...
    para.currentDir, para.imageName, para.exNumber, para.methodName, theta, l1, l2);
saveas(progressWindow, graphImageName)
graphImageName = sprintf('%s/images/%s/%02d/%s/progressWindowRMSE_theta%.3f_lambda1_%.3f_lambda2_%.3f.png', ...
    para.currentDir, para.imageName, para.exNumber, para.methodName, theta, l1, l2);
saveas(progressWindowRMSE, graphImageName)
close(progressWindow);
close(progressWindowRMSE);