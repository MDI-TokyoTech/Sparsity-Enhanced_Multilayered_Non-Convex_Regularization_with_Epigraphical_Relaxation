function [x, time_opt, convergeNum] = LiGME_STV_single_GPU(u_obsv, A, para, u_org, enhance)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Moreau-enhanced STV
% image is dealt as vector data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% u* = argmin_d 1/2|| Psi(u) - u_obsv ||_F^2 + para.mu*Psi_{Bc}(Lcx)
%
% Psi(Lx) = || KPDu ||_2,1
% Psi(Lcx) = Psi(Lx) + iota_[0 1](u)
% Psi_{Bc}(Lcx) = Psi(Lcx) - min_v {Psi(Lcv) + 1/2|| Bc(Lcx - v) ||_F^2}
% L = KPD
% Lc = [KPD;
%        I ]
if exist(sprintf('output/ComputedMatrixData/%s', mfilename), 'dir') == 0
    mkdir(sprintf('output/ComputedMatrixData/%s', mfilename));
end

%-----------------------------------------------------
% Define L and Lc
%-----------------------------------------------------
if exist(sprintf('output/ComputedMatrixData/%s/Lmat_%03dx%03dimage_bsize(%dx%d)_shift(%d.%d).mat',...
        mfilename, para.cols, para.rows, para.blocksize(1), para.blocksize(2),...
        para.shiftstep(1), para.shiftstep(2)), 'file') == 0
    
    D0 = sparse(-speye(para.rows) + circshift( speye(para.rows) , [0,1] )); 
    D0(end,:) = 0; % for zero expanding
    Dv = kron( speye(para.cols) , D0 );

    D0 = sparse(-speye(para.cols) + circshift( speye(para.cols) , [0,1] )); 
    D0(end,:) = 0; % for zero expanding
    Dh = kron( D0 , speye(para.rows) );

    D = cat(1, Dv, Dh);
    D = sparse(kron(eye(3), D));    
    
    if isequal(para.shiftstep, [1,1])
        Ptemp = sparse(prod(para.blocksize)*3*2*(para.cols-para.blocksize(2)+1)*(para.rows-para.blocksize(1)+1), 6*(para.cols*para.rows));
        P_row = 1;
        for j = 1:(para.cols-para.blocksize(2)+1)  % column position
            for i = 1:(para.rows-para.blocksize(1)+1)  % row position
                % left upper point of each patch
                index = para.rows*(j - 1) + i;
        
                for k_dv = 0:2:4
                    for k_col = 0:para.blocksize(2)-1 % columns of karnel
                        for k_row = 0:para.blocksize(1)-1 % para.rows of karnel
                            Ptemp(P_row, index + k_dv*(para.cols*para.rows) + k_col*para.rows +  k_row) = 1;
                            P_row = P_row + 1;
                        end
                    end
                end
                for k_dh = 1:2:5
                    for k_col = 0:para.blocksize(2)-1 % columns of karnel
                        for k_row = 0:para.blocksize(1)-1 % para.rows of karnel
                            Ptemp(P_row, index + k_dh*(para.cols*para.rows) + k_col*para.rows +  k_row) = 1;
                            P_row = P_row + 1;
                        end
                    end
                end
            end
        end
        % Matrix K = 1/prod(para.blocksize);
        KP = Ptemp./prod(para.blocksize);
    
    elseif isequal(para.shiftstep, para.blocksize) % no overlap
        % Ptemp = sparse(2*para.dim*prod(para.blocksize)*(para.rows/para.blocksize(1))*(para.cols/para.blocksize(2)), size(D,1));
        Ptemp = sparse(2*para.dim*para.rows*para.cols, size(D,1));
        P_row = 1;
        for j = 1:para.blocksize(2):(para.cols-para.blocksize(2)+1)  % column position
            for i = 1:para.blocksize(1):(para.rows-para.blocksize(1)+1)  % row position
                % left upper point of each patch
                index = para.rows*(j - 1) + i;
        
                for k_dv = 0:2:4 % Dv block position R=0, G=2, B=4
                    for k_col = 0:para.blocksize(2)-1 % columns of karnel
                        for k_row = 0:para.blocksize(1)-1 % para.rows of karnel
                            Ptemp(P_row, index + k_dv*(para.cols*para.rows) + k_col*para.rows +  k_row) = 1;
                            P_row = P_row + 1;
                        end
                    end
                end
                for k_dh = 1:2:5 % Dh block position R=1, G=3, B=5
                    for k_col = 0:para.blocksize(2)-1 % columns of karnel
                        for k_row = 0:para.blocksize(1)-1 % rows of karnel
                            Ptemp(P_row, index + k_dh*(para.cols*para.rows) + k_col*para.rows +  k_row) = 1;
                            P_row = P_row + 1;
                        end
                    end
                end
            end
        end
        % Matrix K = I;
        KP = Ptemp;
    end
    Lmat = KP*D;
    clear("D0","Dv","Dh","D","Ptemp","KP");
    Lmat = single(full(Lmat));
    
    r = rank(Lmat);
    fprintf("rank(L) = %d\n", r);
    
    % add identity matrix for constraints here
    Lcmat = cat(1, Lmat, eye(size(Lmat, 2), "single"));
    Lc_op2 = max(eig(gather(Lcmat'*Lcmat))); % because eigs function doesn't allow single matrix
    clear("Lcmat");

    save(sprintf('output/ComputedMatrixData/%s/Lmat_%03dx%03dimage_bsize(%dx%d)_shift(%d.%d).mat',...
        mfilename, para.cols, para.rows, para.blocksize(1), para.blocksize(2),...
        para.shiftstep(1), para.shiftstep(2)),...
        "Lc_op2", "r", "Lmat", "-v7.3");
else
    load(sprintf('output/ComputedMatrixData/%s/Lmat_%03dx%03dimage_bsize(%dx%d)_shift(%d.%d).mat',...
        mfilename, para.cols, para.rows, para.blocksize(1), para.blocksize(2),...
        para.shiftstep(1), para.shiftstep(2)));
end

% GPU
Lmat = gpuArray(Lmat);

%-----------------------------------------------------
% Define B
%-----------------------------------------------------
if enhance
    [Bmat, ~] = calculate_B_forSingle(A, Lmat, r, para, "SVD");
    B_op2 = max(eig(Bmat'*Bmat));

    LxShape = size(Lmat*u_obsv(:));
    LxSum = prod(LxShape);

    Bmat = gpuArray(single(full(Bmat))); % GPU
    B = @(z) {reshape(Bmat*reshape(z{1}, [LxSum, 1]), LxShape)};
    Bt = @(z) {reshape(Bmat'*reshape(z{1}, [LxSum, 1]), LxShape)};
else
    B = @(z) {0*z{1}};
    Bt = @(z) {0*z{1}};
    B_op2 = 0;
end

P = @(z) func_PeriodicExpansionGrad(z, para.blocksize, para.shiftstep, para.isTF);
Pt = @(z) func_PeriodicExpansionTransGrad(z, para.isTF); % isTF = 1 makes P a tight frame
switch para.kernel
    case 'Gaussian'
        fkernel = sqrt(fspecial('gaussian', para.blocksize, 0.5));
        fkernel = repmat(fkernel,[fix(para.rows/para.blocksize(1))+1, fix(para.cols/para.blocksize(2))+1, n3, 2, para.blocksize]);
        fkernel = fkernel(1:para.rows,1:para.cols,:,:,:,:);
        fkernel = gpuArray(fkernel);
        K = @(z) z.*fkernel;
        Kt = @(z) z.*fkernel;
    otherwise
        if isequal(para.blocksize, para.shiftstep)
            K = @(z) z; % no overlap
            Kt = @(z) z;
        else
            K = @(z) z/prod(para.blocksize); % all weight is same 
            Kt = @(z) z/prod(para.blocksize);
        end
end

% difference operators
D = @(z) cat(4, z([2:para.rows, para.rows],:,:) - z, z(:,[2:para.cols, para.cols],:)-z);
Dt = @(z) [-z(1,:,:,1); - z(2:para.rows-1,:,:,1) + z(1:para.rows-2,:,:,1); z(para.rows-1,:,:,1)] ...
    +[-z(:,1,:,2), - z(:,2:para.cols-1,:,2) + z(:,1:para.cols-2,:,2), z(:,para.cols-1,:,2)];

%-----------------------------------------------------
% linear operators, proximity operators
%-----------------------------------------------------
Phi = @(z) {A*z{1}};
Phit = @(z) {A'*z{1}};
L = @(z) {K(P(D(z{1})))};
Lt = @(z) {Dt(Pt(Kt(z{1})))};
Lc = @(z) {K(P(D(z{1}))), z{1}}; % add constraint here
Lct = @(z) {Dt(Pt(Kt(z{1}))) + z{2}};

prox_psi1 = cell(1,1); % proximity operator :prox_Psi(L*u)
% prox_psi1{1} = @(z, gamma) gpuArray( Prox_STVnorm_vector(gather(z), gamma, para.blocksize) );
prox_psi1{1} = @(z, gamma) gpuArray(Prox_STVnorm(gather(z), gamma, para.blocksize));
prox_psi2 = cell(1,2); % proximity operator with constraint :prox_Psi(Lc*u)
prox_psi2{1} = prox_psi1{1};
prox_psi2{2} = @(z, gamma) Proj_RangeConstraint(z, [0,1]);


%-----------------------------------------------------
% step size
%-----------------------------------------------------
% [~,S,~] = svds(A'*A);
% A_op2 = max(diag(S));
% clear("S");
A_op2 = eigs(double(gather(A'*A)), 1);

% sigma = ||kappa/2*A'A + mu*Lc'*Lc||_op + (kappa - 1))
sigma = (para.kappa/2*A_op2 + para.mu*Lc_op2) + (para.kappa - 1);

% tau = (kappa/2 + 2/kappa)*mu*||B||_op^2 + (kappa - 1)
tau = (para.kappa/2 + 2/para.kappa)*para.mu*B_op2 + (para.kappa - 1);

%-----------------------------------------------------
% Initialize
%-----------------------------------------------------
y = cell(1, 1);
y{1} = u_obsv;

x = cell(1, 1);
x{1} = u_obsv; % vector

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
N = para.rows*para.cols*para.dim;

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
    x = cellfun(@(z1, z2, z3, z4, z5, z6) z1 - 1/sigma*z2 + para.mu/sigma*z3 - para.mu/sigma*z4 - para.mu/sigma*z5 + 1/sigma*z6, x, Phit(Phi(x)), Lt(Bt(B(L(x)))), Lt(Bt(B(v))), Lct(w), Phit(y), 'UniformOutput', false);
    
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
    RMSE_x(i) = sqrt( sum((x{1} - u_org(:)).^2) / N );

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