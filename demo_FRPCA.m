%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEMO
% Frequency-domain RPCA for shifted signals
% 
% Execute by 4 methods
% (1) NN
% (2) NN in LiGME model
% (3) ASNN
% (4) ASNN in ER-LiGME model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
addpath images
addpath code
addpath code\sub_functions
addpath output

%====================================================
%% Settings
%====================================================
% Original signal
imageFolder = 'images/'; % should include '/'
imageNameList = {'bar'}; % if imageName is 'bar', a 2D signal is generated
imageFormat = 'png';
dim = 1; % max number of color channels of original image

cropSize = 32; % trimming size
cropPoints = {[1,1]};

% Settings for 'bar' image
isShift = true;
shiftStep = 1; % how many pixel shifted per sample
bar_width = floor(cropSize/4);
imageHeight = cropSize;
imageWidth = ceil((cropSize-(bar_width-1)) / shiftStep); % cut empty columns

% Do you load noisy images already generated?
load_noisyImage = false;
% the folder path to load the noisy images
noisyImage_path = "..."; % should be without imageName and extension

% degradation setting
noiseType = 'sparse'; % 3mode: none, sparse, gaussian
noise_sigma = 0.05;
problemType = 'RPCA_none'; % 4mode: none, missing, missingNoiselet, RPCA_none
missrate = 0;         % percentage of missing entries


%-----------------------------------------------------
% definitions for algorithm
%-----------------------------------------------------
% General
methodName = {"NN", "LiGME_NN", "ASNN", "ER_LiGME_ASNN"};
methodPair = {[1,2],[3,4]};
range_lambda1 = cat(2, 0.05, 0.25:0.25:1.25); % for NN and ASNN
range_lambda2 = cat(2, imageHeight*imageWidth*noise_sigma); % for L1ball constraint

% ER-LiGME model
thetaList_1 = [0, 0.1, 0, 1]; % strength of moreau enhancement for each method (static value)
para.kappa = 1.1;            % for setting step size (must be kappa > 1)
para.epsilon_mask = 0; % small elements which less than this value will be 0 in reference data;
extend_weight = 1;
para.extendZ = @(rho) calculate_extendRate(rho, extend_weight);


%-----------------------------------------------------
% training variables
%-----------------------------------------------------
exNumber_max = 10; % repeat number for each image
para.maxiter = 10000; % maximum number of iteration
para.stopcri = 1.0e-5; % stopping criterion
para.problemType = problemType;


%-----------------------------------------------------
% Make directory
%-----------------------------------------------------
% directories to save mat files
Today = string(datetime('today'),'yyyyMMdd');
StartTime = string(datetime('now'),'HHmmss.SSS');
if exist(sprintf('output/%s/%s', mfilename, Today), 'dir') == 0
    mkdir(sprintf('output/%s/%s', mfilename, Today));
end
currentDir = sprintf('output/%s/%s/%s', mfilename, Today, StartTime);
para.currentDir = currentDir;
mkdir(currentDir);
mkdir(strcat(currentDir, "/images"));
mkdir(strcat(currentDir, "/mat"));


%-----------------------------------------------------
% Write memo of paramaters
%-----------------------------------------------------
LogicalStr = {'false', 'true'};
discription = "";
discription = discription + "-- Problem settings --\n";
discription = discription + sprintf("problem type: %s\n", problemType);
discription = discription + sprintf("missing rate: %.2f\n", missrate);
discription = discription + sprintf("noise type: %s\n", noiseType);
discription = discription + sprintf("noise standard deviation: %.2f\n", noise_sigma);
discription = discription + sprintf("range of lambda1 (weight of the regularization term): %s\n", sprintf("%.2f, ",range_lambda1));
discription = discription + sprintf("range of lambda2 (value of L1-ball constraint): %s\n", sprintf("%.2f, ",range_lambda2));
discription = discription + "\n";
discription = discription + "-- LiGME model --\n";
discription = discription + sprintf("theta (NN): %.2f\n", thetaList_1(2));
discription = discription + "\n";
discription = discription + "-- ER-LiGME model --\n";
discription = discription + sprintf("add mask: %s\n", LogicalStr{(para.epsilon_mask ~= 0) + 1});
discription = discription + sprintf("theta (ASNN): %.2f\n", thetaList_1(4));

fileID = fopen(sprintf('%s/discription.txt', currentDir),'w');
fprintf(fileID, discription);
fclose(fileID);


%-----------------------------------------------------
% decleare variables
%-----------------------------------------------------
imageNum = length(imageNameList);
ModifiedImageNameList = cell(1, imageNum);
methodNum = length(methodName);
methodPairNum = length(methodPair);
lambdaNum = length(range_lambda1)*length(range_lambda2);

methodFunc = cell(1,methodNum);
opt_L = zeros(imageHeight, imageWidth, dim, methodNum); % buffer
opt_S = zeros(imageHeight, imageWidth, dim, methodNum); % buffer
diff_opt_L = zeros(imageNum, lambdaNum, methodPairNum, exNumber_max);
Mdiff_opt_L = zeros(imageNum, lambdaNum, methodPairNum);
PSNR_opt = zeros(imageNum, methodNum, lambdaNum, exNumber_max); % optimal image's PSNR
SSIM_opt = zeros(imageNum, methodNum, lambdaNum, exNumber_max); % optimal image's SSIM
PSNR_obsv = zeros(imageNum, exNumber_max);  % noisy image's PSNR
SSIM_obsv = zeros(imageNum, exNumber_max);  % noisy image's SSIM
MPSNR = zeros(imageNum, methodNum, lambdaNum);         % Mean of PSNR for each images 
MSSIM = zeros(imageNum, methodNum, lambdaNum);         % Mean of SSIM for each images 
time_opt = zeros(imageNum, methodNum, lambdaNum, exNumber_max);    % time to complete optimization
convergeNum = zeros(imageNum, methodNum, lambdaNum, exNumber_max); % number of iteration
time_B = zeros(imageNum, methodNum, lambdaNum, exNumber_max);      % time to calculate B


%====================================================
%% Start Experiment
%====================================================
disp("Start Experiment")
for imageID = 1:length(imageNameList)

%-----------------------------------------------------
% Load original image
%-----------------------------------------------------
imageName = imageNameList{imageID};
if strcmp(imageName, 'bar')
    % generate bar image
    u_org = zeros(imageHeight, imageWidth);

    % Shift bar and Generate imageName (example: 'bar16_noShift')
    if isShift
        for k=1:imageWidth
            u_org(1+(k-1)*shiftStep:1+(k-1)*shiftStep+(bar_width-1), k) = 1;
        end
        imageName = strcat(imageName, num2str(imageHeight), '_shift');
    else
        u_org(end-(bar_width-1):end, :) = 1;
        imageName = strcat(imageName, num2str(imageHeight), '_noShift');
    end

else
    % Load images
    u_org = im2double(imread(strcat(imageFolder, imageName, '.', imageFormat)));
end
para.imageName = imageName;
ModifiedImageNameList{imageID} = imageName;

% trimming
cropPoint = cropPoints{imageID};
u_org = u_org(cropPoint(1):cropPoint(1)+imageHeight-1, cropPoint(2):cropPoint(2)+imageWidth-1, :);

[rows, cols, dim] = size(u_org);
para.rows = rows;
para.cols = cols;
para.dim = dim;
N = rows*cols*dim;

% Make directory for save
mkdir(sprintf('%s/images/%s', currentDir, imageName));
mkdir(sprintf('%s/mat/%s', currentDir, imageName));
imwrite(u_org, sprintf('%s/images/%s/org.%s', currentDir, imageName, imageFormat), imageFormat);

%-----------------------------------------------------
% repeat the experements on different noise seed
%-----------------------------------------------------
for exNumber = 1:exNumber_max
para.exNumber = exNumber;

% Make directory for save
mkdir(sprintf('%s/images/%s/%02d', currentDir, imageName, exNumber));
mkdir(sprintf('%s/mat/%s/%02d', currentDir, imageName, exNumber));

%-----------------------------------------------------
% Generate observed image
%-----------------------------------------------------
if ~load_noisyImage
    % Generate noise
    switch noiseType
        case 'none'
            noise = 0;
        case 'sparse'
            % % contains zero values in noise
            % noise = imnoise( zeros(rows, cols, dim) , 'salt & pepper' , noise_sigma );

            % transrate zero to one
            noise = zeros(rows, cols, dim);
            noise(:,:,:) = 0.5;
            noise = imnoise( noise, 'salt & pepper' , noise_sigma );
            noise(noise == 0) = 1;
            noise(noise == 0.5) = 0;
        case 'gaussian'
            noise = noise_sigma*randn(rows, cols, dim);
    end
    
    % Set Phi matrix
    switch problemType
        case 'RPCA_none'
            Phi = @(z) {z{1} + z{2}}; % [I I]
            Phit = @(z) {z{1}, z{1}}; % [I;I]

            % % explicit definition for LiGME
            % A = [speye(N), speye(N)];

            A_op2 = 2; % || A'*A ||_op
            para.A_op2 = A_op2;
    end

    u_obsv = u_org + noise; % low-rank image + sparse noise
    u_obsv = Proj_RangeConstraint(u_obsv, [0, 1]);
    % ---Generate noisy image (end) ---

    save(sprintf('%s/mat/%s/%02d/u_obsv.mat', currentDir, imageName, exNumber), ...
        "u_obsv", "Phi", "Phit", "A_op2", "noise", "noise_sigma");
else
    % Load image
    load(sprintf('%s/%s/%02d/u_obsv.mat', noisyImage_path, imageName, exNumber), ...
        "u_obsv", "Phi", "Phit", "A_op2", "noise", "noise_sigma");
end
PSNR_obsv(imageID, exNumber) = psnr(u_org, u_obsv, 1);
SSIM_obsv(imageID, exNumber) = ssim(u_org, u_obsv, 'DynamicRange', 1);

% Save noisy image
imwrite(u_obsv, sprintf('%s/images/%s/%02d/obsv.%s', currentDir, imageName, exNumber, imageFormat), imageFormat);

% view images
figure
subplot(121), meshz(u_org), zlim([-0.15 1.15]), title('original');
subplot(122), meshz(u_obsv), zlim([-0.15 1.15]), title('observed');


% %--------------------
% % begin GPU
% %--------------------
% disp("move data to GPU")
% u_obsv = gpuArray(u_obsv);
% u_org = gpuArray(u_org);


%--------------------
% define function
%--------------------
methodFunc{1} = @(p, ref) SRPCA_LiGME(u_obsv, p, u_org, false, [0,1]); % not enhanced NN
methodFunc{2} = @(p, ref) SRPCA_LiGME(u_obsv, p, u_org, true, [0,1]); % enhanced NN
methodFunc{3} = @(p, ref) FRPCA_ERLiGME_rangeConstraint(u_obsv, p, u_org, false, [0,1]); % not enhanced ASNN
methodFunc{4} = @(p, ref) FRPCA_ERLiGME_rangeConstraint(u_obsv, p, u_org, true, [0,1], ref(:,:,1,3)); % enhanced ASNN


%====================================================
%% Main loop
%====================================================
disp("Start main loop")
for lambdaID = 1:lambdaNum
    lambda1ID = ceil( lambdaID / length(range_lambda2) );
    lambda2ID = mod(lambdaID, length(range_lambda2)) + 1;
    para.mu = 1;
    para.lambda_1 = range_lambda1( lambda1ID );
    para.lambda_2 = range_lambda2( lambda2ID );

    for methodID = 1:methodNum
        % Make directory for save
        mkdir(sprintf('%s/images/%s/%02d/%s', currentDir, imageName, exNumber, methodName{methodID}));
        mkdir(sprintf('%s/mat/%s/%02d/%s', currentDir, imageName, exNumber, methodName{methodID}));

        % Set paramaters
        para.methodName = methodName{methodID};
        para.theta_1 = thetaList_1(methodID); % streigth of moreau enhancement
        % para.theta_2 = thetaList_2(methodID); % streigth of moreau enhancement
        para.rho_1 = para.theta_1;
        % para.rho_2 = max(para.theta_2, eps);
    
        % optimaization
        [l, s, t, c] = methodFunc{methodID}(para, opt_L);
        disp("Complete to optimize")
        time_opt(imageID,methodID,lambdaID,exNumber) = t;
        convergeNum(imageID,methodID,lambdaID,exNumber) = c;
        opt_L(:,:,1:dim,methodID) = reshape(l, [rows, cols, dim]);
        opt_S(:,:,1:dim,methodID) = reshape(s, [rows, cols, dim]);

        % evaluate
        PSNR_opt(imageID,methodID,lambdaID,exNumber) = psnr(u_org, opt_L(:,:,1:dim,methodID), 1);
        SSIM_opt(imageID,methodID,lambdaID,exNumber) = ssim(u_org, opt_L(:,:,1:dim,methodID), 'DynamicRange', 1);
    
        imwrite(opt_L(:,:,1:dim,methodID), sprintf('%s/images/%s/%02d/%s/lambda1_%.3f_lambda2_%.3f.%s', ...
                currentDir, imageName, exNumber, methodName{methodID}, ...
                para.lambda_1, para.lambda_2, imageFormat), imageFormat);
    end % next method

    % Evalueate difference between enhanced results and not enhanced ones
    for pairID = 1:methodPairNum
        pair = methodPair{pairID};
        N = numel(opt_L(:,:,1:dim, pair(1)));
        % Evaluate by L2 norm
        diff_opt_L(imageID, lambdaID, pairID, exNumber) = ...
            norm( reshape(opt_L(:,:,1:dim,pair(1)), [N, 1]) - reshape(opt_L(:,:,1:dim,pair(2)), [N, 1]) );
    end
end % next lambda
%====================================================

% %--------------------
% % end GPU
% %--------------------
% disp("gather data from GPU")
% u_obsv = gather(u_obsv);
% A = gather(A);
% reset(gpuDevice);

% clear("A");
clear("u_obsv");
close all

end % next experiment (change noise seed)

%--------------------
% mean PSNR of all experiments
%--------------------
for methodID = 1:methodNum
    for lambdaID = 1:lambdaNum
        MPSNR(imageID, methodID, lambdaID) = mean(PSNR_opt(imageID, methodID, lambdaID, :), 4);
        MSSIM(imageID, methodID, lambdaID) = mean(SSIM_opt(imageID, methodID, lambdaID, :), 4);
    end
end

% mean difference
for lambdaID = 1:lambdaNum
    for pairID = 1:methodPairNum
        Mdiff_opt_L(imageID, lambdaID, pairID) = mean(diff_opt_L(imageID, lambdaID, pairID, :), 4);
    end
end

end % change image and continue loop

best_lambda_idx_PSNR = zeros(imageNum, methodNum);
best_lambda_idx_SSIM = zeros(imageNum, methodNum);
for imageID = 1:imageNum
    for i = 1:methodNum
        [~, best_lambda_idx_PSNR(imageID, i)] = max(MPSNR(imageID, i, :));
        [~, best_lambda_idx_SSIM(imageID, i)] = max(MSSIM(imageID, i, :));
    end
end

%--------------------
% save
%--------------------
executeMfilename = mfilename;
resultMatPath = sprintf('%s/mat/all_result.mat', currentDir);
save(resultMatPath, "-v7.3");

% show results
export_results_FRPCA(resultMatPath);