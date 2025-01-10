function [Bmat, time_B] = calculate_B(A, Lmat, r, para, Method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculating B, which is the paramater of LiGME
% paper: A Unified Design of Generalized Moreau Enhancement Matrix for Sparsity Aware LiGME Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist(sprintf('output/ComputedMatrixData/%s', mfilename), 'dir') == 0
    mkdir(sprintf('output/ComputedMatrixData/%s', mfilename));
end

%----------------------------------------------
% Load paramater
%----------------------------------------------
theta = para.theta;
mu = para.mu;

% check files
if exist(sprintf('output/ComputedMatrixData/%s/Bmat_problem(%s)_%s_imsize(%03dx%03d)_%s.mat',...
        mfilename, para.problemType, para.methodName, para.cols, para.rows, Method), 'file') == 0
    %----------------------------------------------
    % create B matrix
    %----------------------------------------------
    if isa(A, 'single')
        errorLimit = 1e-5; % set allowed calculation error
    else
        errorLimit = 1e-8; % set allowed calculation error
    end
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
            [L0, U0, P0, Q0] = lu(gather(Lmat));
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
            if gather(sum(sum(abs(check - sparse(1:r, 1:r, 1, size(Lmat,1), size(Lmat,2)))))) > errorLimit
                error("Failed to calclate P and Q")
            end

        case "SVD"
            %----------------------------------------------
            % fast but not suitable for big matrix
            %----------------------------------------------
            [U, S, Q] = svd(full(gather(Lmat))); % single for big images
            P = pinv(S)*U';
            % if max(size(Lmat)) < 10000
            %     [U, S, Q] = svd(full(Lmat));
            %     P = pinv(S)*U';
            % else
            %     % SVD of sparse matrix
            %     % [U, S, V, flag] = svds(gather(Lmat), r);
            %     % if flag ~= 0
            %     %     error("Failed to calclate P and Q")
            %     % end
            %     [U, S, V] = svd(full(gather(Lmat)), "econ");
            % 
            %     U2 = speye(size(Lmat, 1));
            %     S2 = zeros(size(Lmat));
            %     Q = speye(size(Lmat, 2));
            %     U2(1:size(U,1), 1:size(U,2)) = U;
            %     S2(1:size(S,1), 1:size(S,2)) = S;
            %     Q(1:size(V,1), 1:size(V,2)) = V;
            % 
            %     P = pinv(S2)*U2';
            % end
    end
    % [A1 A2] = A*Q
    AQ = gather( A*Q );
    A1 = AQ(:, 1:r);
    A2 = AQ(:, r+1:end);
    
    % r x r matrix
    MtM = A1'*A1 - A1'*A2*pinv(full(A2))*A1;
    M = A1 - A2*pinv(full(A2))*A1;
    
    % MtM = gather(MtM);
    % M = gather(M);
    if gather( sum(sum(abs(MtM - M'*M))) ) > errorLimit
        error("Failed to calclate M")
    end
    
    % B = [sqrt(theta/mu)*M, O]*P
    Bmat = [sqrt(theta/mu)*M, sparse( zeros(size(M, 1), size(P, 1)-size(M, 2)) )]*P;
    
    time_B = toc
    disp("complete to caluclate B")
    
    P = gather(P);
    M = gather(M);
    save(sprintf('output/ComputedMatrixData/%s/Bmat_problem(%s)_%s_imsize(%03dx%03d)_%s.mat',...
        mfilename, para.problemType, para.methodName, para.cols, para.rows, Method), ...
        "time_B", "P", "M", "-v7.3");
else
    load(sprintf('output/ComputedMatrixData/%s/Bmat_problem(%s)_%s_imsize(%03dx%03d)_%s.mat',...
    mfilename, para.problemType, para.methodName, para.cols, para.rows, Method));

    Bmat = [sqrt(theta/mu)*M, sparse( zeros(size(M, 1), size(P, 1)-size(M, 2)) )]*P;
end