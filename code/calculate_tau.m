function tau = calculate_tau(para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ER-LiGMEのパラメータ計算
% B = [ O O ]
%     [ O sqrt(theta/mu)*I ]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tau = (kappa/2 + 2/kappa)*mu*||B||_op^2 + (kappa - 1)
% ||B||_op = sqrt(theta/mu);
tau = (para.kappa/2 + 2/para.kappa)*para.theta + (para.kappa - 1);