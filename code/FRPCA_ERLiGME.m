function [l, s, time_opt, convergeNum] = FRPCA_ERLiGME(u_obsv, para, u_org, enhance, u_opt)
%==================================
%  input: 2D shifted signal
% output: Low-rank signal, Sparse noise
%==================================
% u = [l, s, z]^T
% Phi = [I I O]
% u* = argmin_u 1/2|| Phi*u - y ||_F^2 + Psi(L*u)
%
% Psi(L*u) = zero(l) + zero(s) + lambda_1*|| z ||_*
% Psi(Lc*u) = Psi(L*u) + iota_{||.||_1 <= lambda_2}(s) + iota_{epi_L2}(Tl, z)
% Psi_B(L*u) = Psi(L*u) - min{Phi(L*v) + 1/2||B(L*u - v)||}
% 
% L = I_3
% Lc = [I O O]
%      [O I O]
%      [O O I]
%      [O I O]
%      [T O O]
%      [O O I]
% T = P[Tr; Ti] (P is permutation matrix)
% B = sqrt(theta/mu)*[O O I]

%% Settings
%-----------------------------------------------------
% step size
%-----------------------------------------------------
l1 = para.lambda_1;
l2 = para.lambda_2;
theta = para.theta_1;
rho = para.rho_1;

Lc_op2 = 1 + 2; % constraint OFF (eigs(max(2I, I+(Tr'Tr + Ti'Ti)), 1))

% sigma = ||kappa/2*A'A + mu*Lc'*Lc||_op + (kappa - 1))
% ||A'A||_op^2 = eigs(AtA'*AtA, 1) = eigs(A'*A, 1)^2
sigma = para.kappa/2*para.A_op2 + para.mu*Lc_op2 + (para.kappa - 1); % for update of x

% tau = (kappa/2 + 2/kappa)*mu*||B||_op^2 + (kappa - 1)
tau = (para.kappa/2 + 2/para.kappa)*theta + (para.kappa - 1); % for update of v

%-----------------------------------------------------
% linear operators, proximity operators
%-----------------------------------------------------
N = para.rows*para.cols*para.dim;

% proximity operator :prox_Psi(L*u)
prox_psi1 = cell(1,3);
prox_psi1{1} = @(z, gamma) z;
prox_psi1{2} = @(z, gamma) z;
prox_psi1{3} = @(z, gamma) ProxNN(z, gamma, l1); % nuclear norm
% proximity operator with constraint :prox_Psi(Lc*u)
prox_psi2 = cell(1,5);
prox_psi2{1} = @(z, gamma) z;
prox_psi2{2} = @(z, gamma) z;
prox_psi2{3} = @(z, gamma) ProxNN(z, gamma, l1); % nuclear norm
prox_psi2{4} = @(z, gamma) ProjL1ball(z, zeros(size(z)), l2);
prox_psi2{5} = @(z1, z2, gamma) Proj_Epigraph_L2_for_ASNN(z1, z2, gamma);
% for color image
if para.dim > 1
    prox_psi1{3} = @(z, gamma) ProxNNcolor(z, gamma*l1);
    prox_psi2{3} = @(z, gamma) ProxNNcolor(z, gamma*l1);
end

Wc = cos(2*pi/para.rows*((0:para.rows-1)')*(0:para.rows-1));
Ws = sin(2*pi/para.rows*((0:para.rows-1)')*(0:para.rows-1));
temp = Wc'*Wc  + (-Ws)'*(-Ws);
scale = 1/sqrt(temp(1,1));
Wc = scale*Wc;
Ws = scale*Ws;
T = @(z) cat(3, Wc*z, -Ws*z);
Tt = @(z) reshape(Wc'*z(:,:,1) - Ws'*z(:,:,2), [para.rows, para.cols]);

L = @(z) z;
Lt = @(z) z;
Lc = @(z) { z{1}, z{2}, z{3}, z{2}, T(z{1}), z{3} };
Lct = @(z) { z{1} + Tt(z{5}), z{2} + z{4}, z{3} + z{6} };
if enhance
    A = @(z) {z{1}+z{2}, sqrt(rho)*z{3}};
    At = @(z) {z{1}, z{1}, sqrt(rho)*z{2}};
    B = @(z) {0*z{1}, 0*z{2}, sqrt(theta/para.mu)*z{3}};
    Bt = @(z) {0*z{1}, 0*z{2}, sqrt(theta/para.mu)*z{3}};

    % add observation data
    z_star = sqrt( sum( T(u_opt).^2 , 3 ) ); % L2norm
    z_star = para.extendZ(rho)*mask_referenceData(z_star, para.epsilon_mask); % add mask to needless components

    y = cell(1, 2);
    y{1} = u_obsv;
    y{2} = sqrt(rho)*z_star;

    x = cell(1, 2);
    x{1} = u_obsv;
    x{2} = zeros(para.rows, para.cols);
    x{3} = zeros(size(y{2}));
else
    % Pure ASNN
    y = cell(1, 1);
    y{1} = u_obsv;

    x = cell(1, 3);
    x{1} = u_obsv;
    x{2} = zeros(para.rows, para.cols);
    x{3} = sqrt( sum( T(u_obsv).^2 , 3 ) );

    A = @(z) { z{1} + z{2} }; % A = [I I O O]
    At = @(z) { z{1}, z{1}, zeros(size(x{3}, 1), size(z{1}, 2)) };
    B = @(z) cellfun(@(zi) 0*zi, z, 'UniformOutput', false);
    Bt = @(z) cellfun(@(zi) 0*zi, z, 'UniformOutput', false);
end

%-----------------------------------------------------
% Initial values
%-----------------------------------------------------
v = L(x);
w = Lc(x);
vnum = numel(v);
wnum = numel(w);
vnumel = numel(v{1});
wnumel = numel(w{1});

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
    for j = 1:4
        w{j} = w{j} - prox_psi2{j}(w{j}, 1);
    end
    [w5temp, w6temp] = prox_psi2{5}(w{5}, w{6}, 1);
    w{5} = w{5} - w5temp;
    w{6} = w{6} - w6temp;

    % check convergence
    converge_rate(i) = norm(reshape(x{1}, [N,1]) - reshape(xpre{1},[N,1])) + ...
        norm(reshape(v{1}, [vnumel,1]) - reshape(vpre{1},[vnumel,1])) + ...
        norm(reshape(w{1}, [wnumel,1]) - reshape(wpre{1},[wnumel,1]));
    fprintf('iter: %d, Error(X) = %f\n', i, converge_rate(i));
    RMSE_L(i) = sqrt( sum(sum(sum((x{1} - u_org).^2))) / N );

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