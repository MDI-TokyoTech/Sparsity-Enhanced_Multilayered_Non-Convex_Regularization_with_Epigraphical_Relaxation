function [x, time_opt, convergeNum] = STV_GPU(u_obsv, Phi, Phit, para, u_org)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% no-enhanced STV
% image is dealt as matrix data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% u* = argmin_d 1/2|| Psi(u) - u_obsv ||_F^2 + para.mu*Psi_{Bc}(Lcx)
%
% Psi(Lx) = || KPDu ||_*,1
% Psi(Lcx) = Psi(Lx) + iota_[0 1](u)
% Psi_{Bc}(Lcx) = Psi(Lcx) - min_v {Psi(Lcv) + 1/2|| Bc(Lcx - v) ||_F^2}
% L = KPD
% Lc = [KPD;
%        I]

%-----------------------------------------------------
% Define L and Lc
%-----------------------------------------------------
blocksize = para.blocksize;
shiftstep = para.shiftstep;
kernel = para.kernel;
isTF = para.isTF;

rows = para.rows;
cols = para.cols;
dim = para.dim;
N = rows*cols*dim;

P = @(z) func_PeriodicExpansionGrad(z, blocksize, shiftstep, isTF);
Pt = @(z) func_PeriodicExpansionTransGrad(z, isTF);
switch kernel
    case 'Gaussian'
        fkernel = sqrt(fspecial('gaussian', blocksize, 0.5));
        fkernel = repmat(fkernel,[fix(rows/blocksize(1))+1, fix(cols/blocksize(2))+1, n3, 2, blocksize]);
        fkernel = fkernel(1:rows,1:cols,:,:,:,:);
        fkernel = gpuArray(fkernel);
        K = @(z) z.*fkernel;
        Kt = @(z) z.*fkernel;
    otherwise
        if isequal(blocksize, shiftstep)
            K = @(z) z; % no overlap
            Kt = @(z) z;
        else
            K = @(z) z/prod(blocksize); % all weight is same 
            Kt = @(z) z/prod(blocksize);
        end
end

% difference operators
D = @(z) cat(4, z([2:rows, rows],:,:) - z, z(:,[2:cols, cols],:)-z);
Dt = @(z) [-z(1,:,:,1); - z(2:rows-1,:,:,1) + z(1:rows-2,:,:,1); z(rows-1,:,:,1)] ...
    +[-z(:,1,:,2), - z(:,2:cols-1,:,2) + z(:,1:cols-2,:,2), z(:,cols-1,:,2)];


%-----------------------------------------------------
% Define B
%-----------------------------------------------------
B = @(z) {0*z{1}};
Bt = @(z) {0*z{1}};


%-----------------------------------------------------
% linear operators, proximity operators
%-----------------------------------------------------
A = @(z) {Phi(z{1})};
At = @(z) {Phit(z{1})};
L = @(z) {K(P(D(z{1})))};
Lt = @(z) {Dt(Pt(Kt(z{1})))};
Lc = @(z) {K(P(D(z{1}))), z{1}}; % add constraint here
Lct = @(z) {Dt(Pt(Kt(z{1}))) + z{2}};

prox_psi1 = cell(1,1); % proximity operator :prox_Psi(L*u)
prox_psi1{1} = @(z, gamma) gpuArray(Prox_STVnorm(gather(z), gamma, blocksize));
prox_psi2 = cell(1,2); % proximity operator with constraint :prox_Psi(Lc*u)
prox_psi2{1} = prox_psi1{1};
prox_psi2{2} = @(z, gamma) Proj_RangeConstraint(z, [0,1]);


%-----------------------------------------------------
% step size
%-----------------------------------------------------
LctLc_op = 9/prod(blocksize); % ||DtD+I||_op <= 9
if isequal(shiftstep, blocksize) % no overlap
    LctLc_op = 9; % ||DtD+I||_op <= 9
end

% sigma = ||kappa/2*A'A + mu*Lc'*Lc||_op + (kappa - 1))
sigma = (para.kappa/2*para.A_op2 + para.mu*LctLc_op) + (para.kappa - 1);

% tau = (kappa/2 + 2/kappa)*mu*||B||_op^2 + (kappa - 1)
tau = para.kappa - 1;

%-----------------------------------------------------
% Initialize
%-----------------------------------------------------
y = cell(1, 1);
y{1} = u_obsv;

x = cell(1, 1);
x{1} = u_obsv;

v = L(x);
w = Lc(x);
vnum = numel(v);
wnum = numel(w);
vnumel = numel(v{1});
wnumel = numel(w{1});

converge_rate = zeros(1, para.maxiter);
RMSE_x = zeros(1, para.maxiter);
progressWindow = figure("Name", mfilename);
progressWindowRMSE = figure("Name", strcat(mfilename, " - RMSE"));
ax = axes("Parent", progressWindow);
ax2 = axes("Parent", progressWindowRMSE);
movegui(progressWindow, 'center');
movegui(progressWindowRMSE, 'east');

%-----------------------------------------------------
% Main Loop
%-----------------------------------------------------
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