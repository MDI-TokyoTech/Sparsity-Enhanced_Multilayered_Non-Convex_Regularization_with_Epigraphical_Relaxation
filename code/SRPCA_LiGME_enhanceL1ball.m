function [l, s, time_opt, convergeNum] = SRPCA_LiGME_enhanceL1ball(u_obsv, para, u_org, enhance, rangeConstraint)
%==================================
%  input: 2D shifted signal
% output: Low-rank signal, Sparse noise
%==================================
% u = [l, s]^T
% Phi = [I I]
% u* = argmin_u 1/2|| Phi*u - y ||_F^2 + mu*Psi(L*u)

% Psi(L*u) = lambda_1*|| l ||_* + iota_{||.||_1 <= lambda_2}(s)
% Psi_B(L*u) = Psi(L*u) - min{Phi(L*v) + 1/2||B(L*u - v)||}
% L = [I O]
%     [O I]
% B = sqrt(theta/mu)*[I I]

% if rangeConstraint ON
%  Psi(Lc*u) = Psi(L*u) + iota_{[a,b]}(l)
% Lc = [I O]
%      [O I]
%      [I O]
%
% if rangeConstraint OFF
%  Psi(Lc*u) = Psi(L*u)
% Lc = [I O]
%      [O I]

%% Settings
l1 = para.lambda_1;
l2 = para.lambda_2;
theta = para.theta_1;


%-----------------------------------------------------
% linear operators, proximity operators
%-----------------------------------------------------
N = para.rows*para.cols*para.dim;
A = @(z) {z{1} + z{2}};
At = @(z) {z{1}, z{1}};
if enhance
    B = @(z) {sqrt(theta/para.mu)*z{1} + sqrt(theta/para.mu)*z{2}};
    Bt = @(z) {sqrt(theta/para.mu)*z{1}, sqrt(theta/para.mu)*z{1}};
    B_op2 = 2*theta/para.mu;
else
    B = @(z) cellfun(@(z1) 0*z1, z, 'UniformOutput', false);
    Bt = @(z) cellfun(@(z1) 0*z1, z, 'UniformOutput', false);
    B_op2 = 0;
end
L = @(z) z;
Lt = @(z) z;

% proximity operator :prox_Psi(L*u)
prox_psi1 = cell(1,2);
prox_psi1{1} = @(z, gamma) ProxNN(z, gamma, l1);
prox_psi1{2} = @(z, gamma) ProjL1ball(z, zeros(size(z)), l2);
% proximity operator with constraint :prox_Psi(Lc*u)
if ~exist('rangeConstraint', 'var')
    prox_psi2 = cell(1,2);
    prox_psi2{1} = @(z, gamma) ProxNN(z, gamma, l1);
    prox_psi2{2} = @(z, gamma) ProjL1ball(z, zeros(size(z)), l2);
    % for glay scale image
    if para.dim > 1
        prox_psi1{1} = @(z, gamma) ProxNNcolor(z, gamma*l1);
        prox_psi2{1} = @(z, gamma) ProxNNcolor(z, gamma*l1);
    end
    Lc = @(z) z;
    Lct = @(z) z;
    LctLc_op2 = 1;
else
    prox_psi2 = cell(1,3);
    prox_psi2{1} = @(z, gamma) ProxNN(z, gamma, l1);
    prox_psi2{2} = @(z, gamma) ProjL1ball(z, zeros(size(z)), l2);
    prox_psi2{3} = @(z, gamma) Proj_RangeConstraint(z, rangeConstraint);
    % for glay scale image
    if para.dim > 1
        prox_psi1{1} = @(z, gamma) ProxNNcolor(z, gamma*l1);
        prox_psi2{1} = @(z, gamma) ProxNNcolor(z, gamma*l1);
    end
    Lc = @(z) {z{1}, z{2}, z{1}};
    Lct = @(z) {z{1}+z{3}, z{2}};
    LctLc_op2 = 2;
end


%-----------------------------------------------------
% step size
%-----------------------------------------------------
% sigma = ||kappa/2*A'A + mu*Lc'*Lc||_op + (kappa - 1))
% ||A'A||_op^2 = eigs(AtA'*AtA, 1) = eigs(A'*A, 1)^2
sigma = para.kappa/2*para.A_op2 + para.mu*LctLc_op2 + (para.kappa - 1); % for update of x

% tau = (kappa/2 + 2/kappa)*mu*||B||_op^2 + (kappa - 1)
tau = (para.kappa/2 + 2/para.kappa)*para.mu*B_op2 + (para.kappa - 1); % for update of v


%-----------------------------------------------------
% Initial values
%-----------------------------------------------------
y = cell(1, 1);
y{1} = u_obsv;

x = cell(1, 2);
x{1} = u_obsv; % l
x{2} = zeros(size(u_obsv)); % s

v = L(x);
w = Lc(x);
vnum = numel(v);
wnum = numel(w);

converge_rate = zeros(1, para.maxiter);
RMSE_L = zeros(1, para.maxiter);
progressWindow = figure("Name", mfilename);
progressWindowRMSE = figure("Name", strcat(mfilename, " - RMSE"));
ax = axes("Parent", progressWindow);
ax2 = axes("Parent", progressWindowRMSE);
movegui(progressWindow, 'center');
movegui(progressWindowRMSE, 'east');

%% main
%-----------------------------------------------------
% main loop
%-----------------------------------------------------
disp(mfilename);
tic;
for i = 1:para.maxiter
    xpre = x;
    vpre = v;
    wpre = w;

    % x_{k+1} = x_pre - 1/sigma*At*A*x_pre + mu/sigma*Lt*Bt*B*L*x_pre -
    %           mu/sigma*Lt*Bt*B*v - mu/sigma*Lct*w + 1/sigma*At*y
    x = cellfun(@(z1, z2, z3, z4, z5, z6) z1 - 1/sigma*z2 + para.mu/sigma*z3 - para.mu/sigma*z4 - para.mu/sigma*z5 + 1/sigma*z6, x, At(A(x)), Lt(Bt(B(L(x)))), Lt(Bt(B(v))), Lct(w), At(y), 'UniformOutput', false);
    
    % v_{k+1} = prox_{mu/tau}ψ[ 2mu/tau(B'BLx_{k+1}) - mu/tau(B'BLx_{k}) + (I - mu/tau(B'B))v ]
    v = cellfun(@(z1, z2, z3, z4) 2*para.mu/tau*z1 - para.mu/tau*z2 + z3 - para.mu/tau*z4, Bt(B(L(x))), Bt(B(L(xpre))), v, Bt(B(v)), 'UniformOutput', false);
    for j = 1:vnum
        v{j} = prox_psi1{j}(v{j}, para.mu/tau);
    end

    % w_{k+1} = 2Lcx_{k+1} - Lcx_{k} + w - prox_{1}ψ[ 2Lcx_{k+1} - Lcx_{k} + w ]
    w = cellfun(@(z1, z2, z3) 2*z1 - z2 + z3, Lc(x), Lc(xpre), w, 'UniformOutput', false); 
    for j = 1:wnum
        w{j} = w{j} - prox_psi2{j}(w{j}, 1);
    end

    % check convergence
    converge_rate(i) = norm(x{1} - xpre{1}) + norm(v{1} - vpre{1}) + norm(w{1} - wpre{1});
    fprintf('iter: %d, Error(X) = %f\n', i, converge_rate(i));
    RMSE_L(i) = sqrt( sum(sum((x{1} - u_org).^2)) / N );

    % view progress
    if mod(i, 100) == 0
        semilogy(ax, 1:i, converge_rate(1:i));
        ylabel(ax, '$\|(x_{n+1},v_{n+1},w_{n+1}) - (x_n,v_n,w_n)\|$','Interpreter','latex')
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
        ylabel(ax, '$\|(x_{n+1},v_{n+1},w_{n+1}) - (x_n,v_n,w_n)\|$','Interpreter','latex')
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