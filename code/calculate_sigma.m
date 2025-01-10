function sigma = calculate_sigma(para, problemType, LctLc_op)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ER-LiGMEのパラメータ計算
% 行列の作用素ノルムは、行列の最大特異値(>=0)
% Aの特異値^2＝A^T*Aの固有値
% A^T*Aの特異値 <= Aの特異値^2
% 
% A = [ Phi O ]
%     [  O  sqrt(rho)*I ]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------
% term1 = || kappa/2*A'A ||_op
%--------------------------------------
switch problemType
    case "none"
        % Phi = I
        % ||Phi||_op = 1
        term1 = para.kappa/2*max(1, para.rho);

    case "missing"
        % Phi is downSamplingMatrix (missing diag matrix)
        % ||Phi||_op = 1
        term1 = para.kappa/2*max(1, para.rho);

    case "missingNoiselet"
        % Phi is downSampling*Noiselet Matrix
        % ||Phi||_op = 1
        %
        %--------------------------------------
        % Code to make sure ||Phi||_op = 1.
        %--------------------------------------
        % N = 256; % rows*cols*dim
        % missrate = 0.5;
        % 
        % % generate test matrix
        % dr = randperm(N)';
        % mesnum =round(N*(1-missrate));
        % OM = dr(1:mesnum);
        % OMind = zeros(sqrt(N), sqrt(N), 3);
        % OMind(OM) = 1;
        % 
        % DownSampling = reshape(OMind, [numel(OMind), 1]);
        % DownSampling = spdiags(DownSampling, 0, N, N);
        % 
        % Noiselet = realnoiselet(eye(N)) / sqrt(N);
        % 
        % A = DownSampling*Noiselet;
        % 
        % [~,D,~] = svd(A'*A);
        % op = max(diag(D))
        %----------------------------------------
        term1 = para.kappa/2*max(1, para.rho);
end

%--------------------------------------
% term2 = ||  mu*Lc'*Lc ||_op
%--------------------------------------
term2 = para.mu*LctLc_op;

% sigma = ||kappa/2*A'A + mu*Lc'*Lc||_op + (kappa - 1))
sigma = term1 + term2 + (para.kappa - 1);