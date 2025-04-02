function [x, time_opt, convergeNum] = ERLiGME_VTV_GPU(u_obsv, Phi, Phit, para, u_org, enhance, u_opt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Moreau-enhanced DVTV with Epigraph relaxation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% u* = argmin_d 1/2|| Psi(u) - u_obsv ||_F^2 + mu*Psi_{Bc}(Lcx)
%
% Psi(x) = Zero(u) + || z ||_1
% Psi(Lcx) = Psi(x) + ι_epi_L2(Du,z) + iota_[0 1](u)
% Psi_{Bc}(Lcx) = Psi(Lcx) - min_v {Psi(v) + 1/2|| Bc(Lcx - v) ||_F^2}
%               = Psi(Lcx) - min||v1||_1

%-----------------------------------------------------
% Definition for Algorithm
%-----------------------------------------------------
% if input is vector, it is reshaped to matrix.
rows = para.rows;
cols = para.cols;
dim = para.dim;
N = rows*cols*dim;
u_obsv = reshape(u_obsv, [rows, cols, dim]);

% difference operators
D = @(z) cat(4, z([2:rows, rows],:,:) - z, z(:,[2:cols, cols],:)-z);
Dt = @(z) [-z(1,:,:,1); - z(2:rows-1,:,:,1) + z(1:rows-2,:,:,1); z(rows-1,:,:,1)] ...
    +[-z(:,1,:,2), - z(:,2:cols-1,:,2) + z(:,1:cols-2,:,2), z(:,cols-1,:,2)];


%-----------------------------------------------------
% Setting proximity operators
%-----------------------------------------------------
% proximity operator
prox_psi1 = cell(1,2);
prox_psi1{1} = @(z, gamma) z;
prox_psi1{2} = @(z, gamma) Prox_L1norm(z, gamma);
% proximity operator with constraint
prox_psi2 = cell(1,4);
prox_psi2{1} = @(z, gamma) z;
prox_psi2{2} = @(z, gamma) Prox_L1norm(z, gamma);
prox_psi2{3} = @(z1, z2) Proj_Epigraph_L2_for_VTV_GPU(z1, z2, 1);
prox_psi2{4} = @(z, gamma) Proj_RangeConstraint(z, [0,1]);


%-----------------------------------------------------
% Setting linear operators
%-----------------------------------------------------
% x~ = [u; z]
% A~ = [ Psi O ]
% L = [ I O ]
%     [ O I ]
% Lc = [ I O ]
%      [ O I ]
%      [ D O ]
%      [ O I ]
%      [ I O ]
% Lct = [ I O Dt O I ]
%       [ O I O I O ]
% B = [ O O ]
%     [ O sqrt(theta/mu)*I ]
% Bc = blkdiag(B, O_3N);

L = @(z) z;
Lt = @(z) z;
Lc = @(z) { z{1}, z{2}, D(z{1}), z{2}, z{1} };
Lct = @(z) { z{1} + Dt(z{3}) + z{5}, z{2} + z{4} };

if enhance
    A = @(z) {Phi(z{1}), sqrt(para.rho)*z{2}};
    At = @(z) {Phit(z{1}), sqrt(para.rho)*z{2}};
    B = @(z) {0*z{1}, sqrt(para.theta/para.mu)*z{2}};
    Bt = @(z) {0*z{1}, sqrt(para.theta/para.mu)*z{2}};

    % add observation data
    Du = D(u_opt);
    z_star = sqrt( sum(Du.^2, 4) );
    z_star = para.extendZ(para.rho)*mask_referenceData(z_star, para.epsilon_mask); % add mask to needless components

    y = cell(1, 2);
    y{1} = u_obsv;
    y{2} = sqrt(para.rho)*z_star;

    x = cell(1, 2);
    x{1} = u_obsv;
    x{2} = zeros(size(y{2}));

    % GPU
    y{2} = gpuArray(y{2});
    x{2} = gpuArray(x{2});
else
    % Pure DVTV
    y = cell(1, 1);
    y{1} = u_obsv;

    x = cell(1, 2);
    x{1} = u_obsv;
    x{2} = zeros(rows, cols, dim, "gpuArray"); % GPU

    A = @(z) {Phi(z{1})}; % A = [Phi O]
    At = @(z) {Phit(z{1}), zeros(size(x{2}), "gpuArray")};
    B = @(z) {0*z{1}, 0*z{2}};
    Bt = @(z) {0*z{1}, 0*z{2}};
end

v = L(x);
w = Lc(x);
vnum = numel(v);
wnum = numel(w);
vnumel = numel(v{1});
wnumel = numel(w{1});

%-----------------------------------------------------
% Step size
%-----------------------------------------------------
% (2-layer)L'L = [2I + D'D, O]
%                [O, 2I]
% ||L'L||_op <= || 2I + D'D ||_op
% max(eig(D'D)) <= 8
% ||L'L||_op <= ( 2 + 8 ) = 10
LctLc_op = 10;

sigma = calculate_sigma(para, para.problemType, LctLc_op);
tau = calculate_tau(para);

%-----------------------------------------------------
% main loop
%-----------------------------------------------------
converge_rate = zeros(1, para.maxiter);
RMSE_x = zeros(1, para.maxiter);
progressWindow = figure("Name", mfilename);
progressWindowRMSE = figure("Name", strcat(mfilename, " - RMSE"));
ax = axes("Parent", progressWindow);
ax2 = axes("Parent", progressWindowRMSE);
movegui(progressWindow, 'center');
movegui(progressWindowRMSE, 'east');

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
    w{1} = w{1} - prox_psi2{1}(w{1}, 1);
    w{2} = w{2} - prox_psi2{2}(w{2}, 1);
    [w3temp, w4temp] = prox_psi2{3}(w{3}, w{4});
    w{3} = w{3} - w3temp;
    w{4} = w{4} - w4temp;
    w{5} = w{5} - prox_psi2{4}(w{5}, 1);

    % check convergence
    converge_rate(i) = norm(reshape(x{1}, [N,1]) - reshape(xpre{1},[N,1])) + ...
        norm(reshape(v{1}, [vnumel,1]) - reshape(vpre{1},[vnumel,1])) + ...
        norm(reshape(w{1}, [wnumel,1]) - reshape(wpre{1},[wnumel,1]));
    fprintf('iter: %d, Error(X) = %f\n', i, converge_rate(i));
    RMSE_x(i) = sqrt( sum(sum(sum((x{1} - u_org).^2))) / N );

    % view progress
    if mod(i, 50) == 0
        semilogy(ax, 1:i, converge_rate(1:i));
        ylabel(ax, '$\|(x_{n+1},v_{n+1},w_{n+1}) - (x_n,v_n,w_n)\|$','Interpreter','latex')
        xlabel(ax, 'iteration')
        title(ax,'converge rate')

        semilogy(ax2, 1:i, RMSE_x(1:i));
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

        semilogy(ax2, 1:i, RMSE_x(1:i));
        ylabel(ax2, 'RMSE','Interpreter','latex')
        xlabel(ax2, 'iteration')
        title(ax2,'converge rate')
        drawnow
        break;
    end
end
time_opt = toc;
convergeNum = i;
x = gather(x);

save(sprintf('%s/mat/%s/%02d/%s/mu%.3f_theta%.3f.mat', para.currentDir, para.imageName,...
    para.exNumber, para.methodName, para.mu, para.theta), ...
    "x", "converge_rate", "convergeNum", "time_opt", "RMSE_x", "para", "sigma", "tau");
%-----------------------------------------------------
% Save progress
%-----------------------------------------------------
graphImageName = sprintf('%s/images/%s/%02d/%s/progressWindow_mu%.3f_theta%.3f.png', ...
    para.currentDir, para.imageName, para.exNumber, para.methodName, para.mu, para.theta);
saveas(progressWindow, graphImageName)
graphImageName = sprintf('%s/images/%s/%02d/%s/progressWindowRMSE_mu%.3f_theta%.3f.png', ...
    para.currentDir, para.imageName, para.exNumber, para.methodName, para.mu, para.theta);
saveas(progressWindowRMSE, graphImageName)
close(progressWindow);
close(progressWindowRMSE);