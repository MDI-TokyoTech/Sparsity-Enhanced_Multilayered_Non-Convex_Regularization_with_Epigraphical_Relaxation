function [Bmat, time_B] = calculate_B_forSingle(A, Lmat, r, para, Method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculating B, which is the paramater of LiGME
% paper: A Unified Design of Generalized Moreau Enhancement Matrix for Sparsity Aware LiGME Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------
% Load paramater
%----------------------------------------------
theta = para.theta;
mu = para.mu;

% check files
if exist(sprintf('%s/mat/%s/%02d/%s/Bmat_mu%.3f_theta%.3f_%s.mat',...
        para.currentDir, para.imageName, para.exNumber, para.methodName, ...
        para.mu, para.theta, Method), 'file') == 0
    %----------------------------------------------
    % create B matrix
    %----------------------------------------------
    % errorLimit = 1e-5; % set allowed calculation error
    errorLimit = 1e-4; % set allowed calculation error (for more than 63 x 63 image)
    disp("start to caluclate B")
    tic
    
    switch Method
        case "RREF"
            %----------------------------------------------
            % very slow but accurate method
            %----------------------------------------------
            L1 = sparse(zeros(size(Lmat, 1)));
            L1(:, 1:size(Lmat,2)) = Lmat;
            L1E = [L1 speye(size(L1))];
            fprintf("caluclating RREF of L1...\n");
            L1Erref = rref(L1E);
        
            P = L1Erref( : , size(L1E,1)+1:end );
            L2 = L1Erref( : , 1:size(Lmat,2) );
            L2E = [ L2' speye( size(L2 , 2) ) ];
            fprintf("caluclating RREF of L2...\n");
            L2Erref = rref(L2E);
        
            Q = (L2Erref( : , end-size(L2Erref,1)+1:end ))';
    
            % P*L*Q = [I_{r} O; O O] --------------
            % P and Q are calclated correctly?
            check = P*Lmat*Q;
            if sum(sum(abs(check - sparse(1:r, 1:r, 1, size(Lmat,1), size(Lmat,2))))) > errorLimit
                error("Failed to calclate P and Q")
            end
        case "LDU"
            %----------------------------------------------
            % fast but Error occuer when "L1_color_withD", "STV"
            %----------------------------------------------
            [L0, U0, P0, Q0] = lu(Lmat);
            [l, n] = size(Lmat);
        
            % convert to square matrix
            Lsqu = speye(l);
            Lsqu(:, 1:r) = L0(:, 1:r);
        
            % convert to square matrix
            Usqu = speye(n);
            Usqu(1:r, :) = U0(1:r, :);
            
            % P*L*Q = Lsqu*[I_{r} O; O O]*Usqu
            P = inv(P0'*Lsqu);
            Q = inv(Usqu*Q0');
    
            if anynan(P) || anynan(Q)
                error("Failed to calclate inverse matrix")
            end
    
            % P*L*Q = [I_{r} O; O O] --------------
            % P and Q are calclated correctly?
            check = P*Lmat*Q;
            if sum(sum(abs(check - sparse(1:r, 1:r, 1, size(Lmat,1), size(Lmat,2))))) > errorLimit
                error("Failed to calclate P and Q")
            end
        case "SVD"
            %----------------------------------------------
            % fast but not suitable for big matrix
            %----------------------------------------------
            [U, S, Q] = svd(single(full(gather(Lmat)))); % single for big images
            P = pinv(S)*U';
    end
    % [A1 A2] = A*Q
    AQ = single(full(A))*Q;
    A1 = AQ(:, 1:r);
    A2 = AQ(:, r+1:end);
    
    % r x r matrix
    MtM = A1'*A1 - A1'*A2*pinv(full(A2))*A1;
    M = A1 - A2*pinv(full(A2))*A1;
    
    MtM = gather(MtM);
    M = gather(M);
    errorMax = max(max(abs(MtM - M'*M)));
    if errorMax > errorLimit
        error("Failed to calclate M")
    end
    
    % B = [sqrt(theta/mu)*M, O]*P
    Bmat = [sqrt(theta/mu)*M, zeros(size(M, 1), size(P, 1)-size(M, 2))]*P;
    
    time_B = toc
    disp("complete to caluclate B")
    
    save(sprintf('%s/mat/%s/%02d/%s/Bmat_mu%.3f_theta%.3f_%s.mat',...
        para.currentDir, para.imageName, para.exNumber, para.methodName, ...
        para.mu, para.theta, Method), "time_B", "P", "M", "-v7.3");
else
    load(sprintf('%s/mat/%s/%02d/%s/Bmat_mu%.3f_theta%.3f_%s.mat',...
        para.currentDir, para.imageName, para.exNumber, para.methodName, ...
        para.mu, para.theta, Method));

    Bmat = [sqrt(theta/mu)*M, zeros(size(M, 1), size(P, 1)-size(M, 2))]*P;
end