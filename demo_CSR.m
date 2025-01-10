%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEMO
% Compressed image sensing reconstruction
% 
% Execute 6 methods
% (1) DVTV
% (2) DVTV in ER-LiGME model
% (3) STV
% (4) STV in ER-LiGME model
% (5) DSTV
% (6) DSTV in ER-LiGME model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
addpath images
addpath code
addpath code\sub_functions
addpath output

%====================================================
%% Settings
%====================================================
% original image
imageFolder = 'images/BSDS256/color/'; % should include '/'
imageNameList = {'BSDS21c'}; % {'BSDS1c', 'BSDS2c', ... }
imageFormat = 'png';
dim = 3; % the number of color channels of original image

cropSize = 64; % trimming size (square)
cropPoints = {[99,99]}; % trimming point (left upper corner) {[y,x], [y,x], ... }

% Do you load noisy images already generated?
load_noisyImage = false;
% the folder path to load the noisy images
noisyImage_path = "..."; % should be without imageName and extension

% degradation setting
noiseType = 'gaussian'; % 3mode: none, sparse, gaussian
noise_sigma = 0.1;
problemType = 'missingNoiselet'; % 3mode: none, missing, missingNoiselet
missrate = 0.5; % percentage of missing entries


%-----------------------------------------------------
% definitions for algorithm
%-----------------------------------------------------
% General
methodName = {"DVTV", "ER_LiGME_DVTV", "STV", "ER_LiGME_STV", "DSTV", "ER_LiGME_DSTV"};
range_mu = cat(2, 0.05, 0.1:0.1:0.5);

% STV, DSTV
para.blocksize = [3,3];  % block size of STV
para.shiftstep = [1,1];  % [1,1] means full overlap
para.kernel = 'Uniform'; % weighting for each block
para.isTF = 1;           % isTF = 1 makes P a tight frame
para.wlumi = 0.5;        % weight of luminance variation

% ER-LiGME model
thetaList = [0, 4, 0, 4, 0, 4]; % strength of moreau enhancement for each method
para.kappa = 1.01;       % for setting step size (must be kappa > 1)
para.epsilon_mask = 0;   % small elements which less than this value will be 0 in reference data;
para.extendZ = @(rho) calculate_extendRate(rho, 1); % extend rate of reference data


%-----------------------------------------------------
% training variables
%-----------------------------------------------------
exNumber_max = 1; % number of repetitions with a different noise seed applied to each image.
para.maxiter = 30000; % maximum number of iteration
para.stopcri = 1.0e-3; % stopping criterion
para.problemType = problemType;

useGPU = true; % true: use GPU, false: use CPU to execute.
try
    gpu = gpuDevice();
    disp('GPU is available:');
    disp(gpu.Name);
catch
    disp('No GPU available or GPU is not supported.');
    useGPU = false;
end

% %-----------------------------------------------------
% % Overwhite settings for debug
% %-----------------------------------------------------
% exNumber_max = 1; % number of repeat for each image
% para.stopcri = 1.0e-2; % stopping criterion
% imageNameList = {'BSDS21c','BSDS30c'};
% cropSize = 16; % trimming size (square) *should be power of 3 for STV
% cropPoints = {[5,67],[80,58]}; % trimming point (left upper corner)


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
discription = discription + sprintf("range of mu (weight of the regularization term): %s\n", sprintf("%.2f, ",range_mu));
% discription = discription + "\n";
% discription = discription + "-- LiGME model --\n";
% discription = discription + sprintf("theta (DVTV): %.2f\n", thetaList(2));
% discription = discription + sprintf("theta (STV) : %.2f\n", thetaList(2));
discription = discription + "\n";
discription = discription + "-- ER-LiGME model --\n";
discription = discription + sprintf("add mask: %s\n", LogicalStr{(para.epsilon_mask ~= 0) + 1});
discription = discription + sprintf("theta (DVTV): %.2f\n", thetaList(2));
discription = discription + sprintf("theta (STV) : %.2f\n", thetaList(4));
discription = discription + sprintf("theta (DSTV): %.2f\n", thetaList(6));

fileID = fopen(sprintf('%s/discription.txt', currentDir),'w');
fprintf(fileID, discription);
fclose(fileID);


%-----------------------------------------------------
% decleare variables
%-----------------------------------------------------
imageNum = length(imageNameList);
methodNum = length(methodName);
muNum = length(range_mu);

methodFunc = cell(1,methodNum);
opt_x = zeros(cropSize, cropSize, dim, methodNum); % buffer for ER-LiGME
PSNR_opt = zeros(imageNum, methodNum, muNum, exNumber_max); % optimal image's PSNR
SSIM_opt = zeros(imageNum, methodNum, muNum, exNumber_max); % optimal image's SSIM
PSNR_obsv = zeros(imageNum, exNumber_max);  % noisy image's PSNR
SSIM_obsv = zeros(imageNum, exNumber_max);  % noisy image's SSIM
MPSNR = zeros(imageNum, methodNum, muNum);         % Mean of PSNR for each images 
MSSIM = zeros(imageNum, methodNum, muNum);         % Mean of SSIM for each images 
time_opt = zeros(imageNum, methodNum, muNum, exNumber_max);    % time to complete optimization
convergeNum = zeros(imageNum, methodNum, muNum, exNumber_max); % number of iteration
time_B = zeros(imageNum, methodNum, muNum, exNumber_max);      % time to calculate B


%====================================================
%% Start Experiment
%====================================================
for imageID = 1:length(imageNameList)
disp("loading original image...")

imageName = imageNameList{imageID};
para.imageName = imageName;

% Make directory to save
mkdir(sprintf('%s/images/%s', currentDir, imageName));
mkdir(sprintf('%s/mat/%s', currentDir, imageName));

%-----------------------------------------------------
% Load original image
%-----------------------------------------------------
disp("Load original image")
u_org = im2double(imread(strcat(imageFolder, imageName, '.', imageFormat))); % normalize 0 - 255 to 0 - 1
% trimming
cropPoint = cropPoints{imageID};
u_org = u_org(cropPoint(1):cropPoint(1)+cropSize-1, cropPoint(2):cropPoint(2)+cropSize-1, :);
% save
imwrite(u_org, sprintf('%s/images/%s/org.%s', currentDir, imageName, imageFormat), imageFormat);

[rows, cols, dim] = size(u_org);
para.rows = rows;
para.cols = cols;
para.dim = dim;
N = rows*cols*dim; % number of elements

%-----------------------------------------------------
% repeat the experements on different noise seed
%-----------------------------------------------------
for exNumber = 1:exNumber_max
disp("Start experiment")
para.exNumber = exNumber;

% Make directory to save
mkdir(sprintf('%s/images/%s/%02d', currentDir, imageName, exNumber));
mkdir(sprintf('%s/mat/%s/%02d', currentDir, imageName, exNumber));

%-----------------------------------------------------
% Generate observed image
%-----------------------------------------------------
disp("Ganerating noisy image...")
if ~load_noisyImage
    % Generate noise
    switch noiseType
        case 'none'
            noise = 0;
        case 'sparse'
            noise_ex = -0.5 + (0.5 - (-0.5))*rand(rows, cols, dim); % matrix of extend rate
            noise = noise_ex.*imnoise( zeros(rows, cols, dim) , 'salt & pepper' , noise_sigma );
        case 'gaussian'
            noise = noise_sigma*randn(rows, cols, dim);
    end

    % Set Phi matrix
    switch problemType
    case 'none'
        % % explicit definition for LiGME
        % A = speye(N);
        Phi = @(z) z;
        Phit = @(z) z;

        A_op2 = 1; % || A'*A ||_op
        para.A_op2 = A_op2;
    case 'missing'
        dr = randperm(N)';
        mesnum =round(N*(1-missrate));
        OM = dr(1:mesnum);
        OMind = zeros(rows, cols, dim);
        OMind(OM) = 1;

        % % explicit definition for LiGME
        % A = reshape(OMind, [numel(OMind), 1]);
        % A = spdiags(A, 0, N, N);

        Phi = @(z) z.*OMind;
        Phit = @(z) z.*OMind;

        noise = noise.*OMind;

        A_op2 = 1; % || A'*A ||_op
        para.A_op2 = A_op2;
    case 'missingNoiselet'
        dr = randperm(N)';
        mesnum =round(N*(1-missrate));
        OM = dr(1:mesnum);
        OMind = zeros(rows, cols, dim);
        OMind(OM) = 1;
        
        % % explicit definition for LiGME
        % DownSamplingMatrix = reshape(OMind, [numel(OMind), 1]);
        % DownSamplingMatrix = spdiags(DownSamplingMatrix, 0, N, N);
        % Noiseletmatrix = realnoiselet(eye(rows*cols)) ./ sqrt(rows*cols);
        % if dim > 1 % color image
        %     Noiseletmatrix = blkdiag(Noiseletmatrix, Noiseletmatrix, Noiseletmatrix);
        % end
        % A = DownSamplingMatrix*Noiseletmatrix;

        if useGPU
            Phi = @(z) gpuArray( func_Noiselet(gather(z)).*OMind );
            Phit = @(z) gpuArray( func_NoiseletTrans(gather(z.*OMind)) );
        else
            Phi = @(z) func_Noiselet(z).*OMind;
            Phit = @(z) func_NoiseletTrans(z.*OMind);
        end

        noise = noise.*OMind;

        A_op2 = 1; % || A'*A ||_op
        para.A_op2 = A_op2;
    end

    u_obsv = Phi(u_org) + noise;
    % ---Generate noisy image (end) ---

    save(sprintf('%s/mat/%s/%02d/u_obsv.mat', currentDir, imageName, exNumber), ...
        "u_obsv", "Phi", "Phit", "A_op2", "noise", "noise_sigma");

    if strcmp(problemType,'missing') || strcmp(problemType,'missingNoiselet')
        save(sprintf('%s/mat/%s/%02d/u_obsv.mat', currentDir, imageName, exNumber), ...
        "OMind", "-append");
    end
else
    % Load image
    load(sprintf('%s/%s/%02d/u_obsv.mat', noisyImage_path, imageName, exNumber), ...
        "u_obsv", "Phi", "Phit", "A_op2", "noise", "noise_sigma");

    if strcmp(problemType,'missing') || strcmp(problemType,'missingNoiselet')
        load(sprintf('%s/%s/%02d/u_obsv.mat', noisyImage_path, imageName, exNumber), ...
            "OMind");
    end
end
PSNR_obsv(imageID, exNumber) = psnr(u_org, gather(Phit(u_obsv)), 1);
SSIM_obsv(imageID, exNumber) = ssim(u_org, gather(Phit(u_obsv)), 'DynamicRange', 1);

% Save images
imwrite( gather(Phit(u_obsv)), sprintf('%s/images/%s/%02d/obsv.%s', currentDir, imageName, exNumber, imageFormat), imageFormat);

% view images
figure
subplot(121), imshow(u_org), title('original');
subplot(122), imshow(gather(Phit(u_obsv))), title('observed');

if useGPU
    %--------------------
    % begin GPU
    %--------------------
    disp("Loading GPU device...")
    u_obsv = gpuArray(u_obsv);
    % A = gpuArray(A);
    if strcmp(problemType,'missing') || strcmp(problemType,'missingNoiselet')
        OMind = gpuArray(OMind);
    end

    %--------------------
    % define function
    %--------------------
    methodFunc{1} = @(p, ref) DVTV_GPU(u_obsv, Phi, Phit, p, u_org);
    methodFunc{2} = @(p, ref) ERLiGME_DVTV_GPU(u_obsv, Phi, Phit, p, u_org, true, ref(:,:,:,1));
    methodFunc{3} = @(p, ref) STV_GPU(u_obsv, Phi, Phit, p, u_org);
    methodFunc{4} = @(p, ref) ERLiGME_STV_GPU(u_obsv, Phi, Phit, p, u_org, true, ref(:,:,:,3));
    methodFunc{5} = @(p, ref) ERLiGME_DSTV_GPU(u_obsv, Phi, Phit, p, u_org, false);
    methodFunc{6} = @(p, ref) ERLiGME_DSTV_GPU(u_obsv, Phi, Phit, p, u_org, true, ref(:,:,:,5));
else
    % use CPU
    %--------------------
    % define function
    %--------------------
    methodFunc{1} = @(p, ref) DVTV_CPU(u_obsv, Phi, Phit, p, u_org);
    methodFunc{2} = @(p, ref) ERLiGME_DVTV_CPU(u_obsv, Phi, Phit, p, u_org, true, ref(:,:,:,1));
    methodFunc{3} = @(p, ref) STV_CPU(u_obsv, Phi, Phit, p, u_org);
    methodFunc{4} = @(p, ref) ERLiGME_STV_CPU(u_obsv, Phi, Phit, p, u_org, true, ref(:,:,:,3));
    methodFunc{5} = @(p, ref) ERLiGME_DSTV_CPU(u_obsv, Phi, Phit, p, u_org, false);
    methodFunc{6} = @(p, ref) ERLiGME_DSTV_CPU(u_obsv, Phi, Phit, p, u_org, true, ref(:,:,:,5));
end

%====================================================
%% Main loop
%====================================================
disp("Start main loop")
for muID = 1:muNum
for methodID = 1:methodNum
    % make directory to save
    mkdir(sprintf('%s/images/%s/%02d/%s', currentDir, imageName, exNumber, methodName{methodID}));
    mkdir(sprintf('%s/mat/%s/%02d/%s', currentDir, imageName, exNumber, methodName{methodID}));

    % set paramaters for each method
    para.methodName = methodName{methodID};
    para.mu = range_mu(muID);        % weight of penalty function
    para.theta = thetaList(methodID);% strength of moreau enhancement
    para.rho = max(eps, para.theta); % weight of z* (non zero)

    % optimaization
    [x, time_opt(imageID,methodID,muID,exNumber), convergeNum(imageID,methodID,muID,exNumber)] = methodFunc{methodID}(para, opt_x);
    opt_x(:,:,:,methodID) = reshape(x{1}, [rows, cols, dim]);
    disp("complete to optimize")
    PSNR_opt(imageID,methodID,muID,exNumber) = psnr(u_org, opt_x(:,:,:,methodID), 1);
    SSIM_opt(imageID,methodID,muID,exNumber) = ssim(u_org, opt_x(:,:,:,methodID), 'DynamicRange', 1);

    % % view
    % figure
    % imshow(opt_x(:,:,:,methodID))
    % title(sprintf("opt%d - %s", methodID, methodName{methodID}), "Interpreter","none")

    % save
    imwrite(opt_x(:,:,:,methodID), sprintf('%s/images/%s/%02d/%s/mu%.3f.%s', ...
            currentDir, imageName, exNumber, methodName{methodID}, ...
            para.mu, imageFormat), imageFormat);
end % next method
end % next mu
%====================================================

if useGPU
    %--------------------
    % end GPU
    %--------------------
    disp("gather data from GPU")
    u_obsv = gather(u_obsv);
    if strcmp(problemType,'missing') || strcmp(problemType,'missingNoiselet')
        OMind = gather(OMind);
    end
    % reset(gpuDevice);
end

close all

end % next experiment (change noise)

%--------------------
% mean PSNR of all experiments
%--------------------
for methodID = 1:methodNum
    for muID = 1:muNum
        MPSNR(imageID, methodID, muID) = mean(PSNR_opt(imageID, methodID, muID, :), 4);
        MSSIM(imageID, methodID, muID) = mean(SSIM_opt(imageID, methodID, muID, :), 4);
        % fprintf('---opt%d %s PSNR: %f\n', methodID, methodName{methodID}, MPSNR(imageID, i));
    end
end

end % change image and continue loop

best_mu_idx_PSNR = zeros(imageNum, methodNum);
best_mu_idx_SSIM = zeros(imageNum, methodNum);
for imageID = 1:imageNum
    for i = 1:methodNum
        [~, best_mu_idx_PSNR(imageID, i)] = max(MPSNR(imageID, i, :));
        [~, best_mu_idx_SSIM(imageID, i)] = max(MSSIM(imageID, i, :));
    end
end

%--------------------
% save
%--------------------
% clear("A"); % Because A is restorable by OMind, and it has very huge size
executeMfilename = mfilename;
resultMatPath = sprintf('%s/mat/all_result.mat', currentDir);
save(resultMatPath, "-v7.3");

% show results
export_results_CSR(resultMatPath);