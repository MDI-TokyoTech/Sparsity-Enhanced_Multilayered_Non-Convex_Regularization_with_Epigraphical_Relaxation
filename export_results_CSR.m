%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show and export results of DEMO
%
% input: path to all_result.mat (string)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function export_results_CSR(matFilePath)

%====================================================
% User settings
%====================================================
shown_exNumID = 1;
shown_muID = -1; % when -1, shown_muID is set to best muID for ER-LiGME DSTV

% Graph of PSNR/SSIM
imsize_PSNR = [480 240];
imsize_SSIM = imsize_PSNR;

% Histgram of abs(Dx)
nbins = 25;
Dcri = 1; % % cut abs(Dx) less than Dcri
imsize_Hist = [140 100];


%====================================================
% Load
%====================================================
if exist("matFilePath", "var")
    load(matFilePath);

    addpath code\sub_functions
    Phit = @(z) func_NoiseletTrans(double(gather(z.*OMind)));
end
sourceDir = para.currentDir; % input folder name
destDir = executeMfilename; % output folder name
timeStr = extractBetween(sourceDir, strlength(sourceDir)-9, strlength(sourceDir));
dayStr = extractBetween(sourceDir, strlength(sourceDir)-18, strlength(sourceDir)-11);
outDir = sprintf("output/%s/%s/%s", destDir, dayStr{1}, timeStr{1});
if ~exist(sprintf("%s/figs", outDir), "dir")
    mkdir(sprintf("%s/figs", outDir));
end

dim = 3; % Max value of dimension
u_org_buf = zeros(cropSize, cropSize, dim, imageNum); % buffer
u_obsv_buf = zeros(cropSize, cropSize, dim, imageNum, exNumber_max); % buffer
u_opt_buf = zeros(cropSize, cropSize, dim, imageNum, methodNum, muNum, exNumber_max); % buffer

for imageID = 1:imageNum
    fprintf("Loading %s...\n", imageNameList{imageID});

    if ~exist(sprintf("%s/images/%s", outDir, imageNameList{imageID}), "dir")
        mkdir(sprintf("%s/images/%s", outDir, imageNameList{imageID}));
    end

    u_org = im2single(imread(strcat(imageFolder, imageNameList{imageID}, '.', imageFormat)));
    cropPoint = cropPoints{imageID};
    u_org_buf(:,:,:,imageID) = u_org(cropPoint(1):cropPoint(1)+cropSize-1, cropPoint(2):cropPoint(2)+cropSize-1, :); % trimming

    for exNumber = 1:exNumber_max
        load(sprintf("%s/mat/%s/%02d/u_obsv.mat", sourceDir, imageNameList{imageID},...
            exNumber), "-mat", "u_obsv");
        u_obsv_buf(:,:,:,imageID,exNumber) = reshape(gather(u_obsv), [cropSize, cropSize, dim]);

        for methodID = 1:methodNum
            for muID = 1:muNum
                load(sprintf("%s/mat/%s/%02d/%s/mu%.3f_theta%.3f.mat", sourceDir,...
                    imageNameList{imageID}, exNumber, methodName{methodID},...
                    range_mu(muID), thetaList(methodID)), "-mat", "x");
                u_opt = reshape(gather(x{1}), [cropSize, cropSize, dim]);
                u_opt_buf(:,:,:,imageID,methodID,muID,exNumber) = u_opt;
            end
        end
    end
end

if shown_muID == -1
    shown_muID = median(best_mu_idx_PSNR(:, 6));
end
fprintf("shown results for mu = %.3f\n", range_mu(shown_muID));

%--------------------
% save max(MPSNR, MSSIM) to Excel file
%--------------------
methodNameModified = cell(size(methodName, 2), 1);
for i = 1:length(methodName)
    methodNameModified{i} = strrep(methodName{i}, '_', ' ');
    methodNameModified{i} = strrep(methodNameModified{i}, 'ER LiGME', 'ER-LiGME');
end

excelFilePath = sprintf('%s/evaluation.xlsx', outDir);

varNames = ["mu", "method", "PSNR", "SSIM"];
varTypes = cell(1, size(varNames, 2));
varTypes(1) = {"double"};
varTypes(2) = {"string"};
varTypes(3:end) = {"double"};
sz = [muNum * methodNum, size(varNames, 2)];

for inputID = 1:imageNum
    T = table('Size',sz,'VariableTypes',string(varTypes),'VariableNames',varNames);

    for muID = 1:muNum
        muRow = 1+(muID-1) * methodNum;
        PSNRcolumn = MPSNR(inputID, :, muID);
        SSIMcolumn = MSSIM(inputID, :, muID);
    
        for methodID = 1:methodNum
            T(muRow+(methodID-1), 1) = num2cell( range_mu(muID) );
            T(muRow+(methodID-1), 2) = { methodName{methodID} };
            T(muRow+(methodID-1), 3) = num2cell( PSNRcolumn(methodID) );
            T(muRow+(methodID-1), 4) = num2cell( SSIMcolumn(methodID) );
        end
    end % mu

    writetable(T, excelFilePath, 'Sheet', imageNameList{inputID});
end


%====================================================
%% view
%====================================================
for imageID = 1:imageNum
    PSNRgraphs = figure("Name", imageNameList{imageID});
    PSNRgraphs.Position(3:4) = imsize_PSNR;

    tiledlayout(PSNRgraphs, 1, 1, 'TileSpacing','Compact', 'Padding', 'Compact');
    nexttile;
    hold on
    lineWidth = 0.5;
    for methodID = 1:methodNum
        switch methodID
            case 1
                font = "k:";
                mark = "o";
            case 2
                font = "r:";
                mark = "o";
            case 3
                font = "k-.";
                mark = "^";
            case 4
                font = "r-.";
                mark = "^";
            case 5
                font = "k";
                mark = "pentagram";
            case 6
                font = "r";
                mark = "pentagram";
                lineWidth = 1;
        end
        plot(range_mu, squeeze(MPSNR(imageID, methodID, :)), font, "Marker", mark, "LineWidth", lineWidth);
    end
    hold off
    xlabel("\mu")
    ylabel("PSNR")
    % ylim([15, 36]);
    % title(imageNameList{imageID});

    % legend(ax, methodName,'interpreter','none', 'Location', 'best')
    lgd = legend(methodNameModified,'interpreter','none');
    lgd.Layout.Tile = 'west';

    graphImageName = sprintf('%s/images/%s/PSNR.png', outDir, imageNameList{imageID});
    saveas(PSNRgraphs, graphImageName)
    graphImageName = sprintf('%s/figs/%s_PSNR.fig', outDir, imageNameList{imageID});
    saveas(PSNRgraphs, graphImageName)
    
    close(PSNRgraphs);
end

f = figure("Name", "PSNR");
tiledlayout(f, 1, imageNum, 'TileSpacing','Compact', 'Padding', 'Compact');
f.Position(3) = f.Position(3)*imageNum;
f.Position(3:4) = f.Position(3:4)*0.75;
for imageID = 1:imageNum
    nexttile
    hold on
    lineWidth = 0.5;
    for methodID = 1:methodNum
        switch methodID
            case 1
                font = "k:";
                mark = "o";
            case 2
                font = "r:";
                mark = "o";
            case 3
                font = "k-.";
                mark = "^";
            case 4
                font = "r-.";
                mark = "^";
            case 5
                font = "k";
                mark = "pentagram";
            case 6
                font = "r";
                mark = "pentagram";
                lineWidth = 1;
        end
        plot(range_mu, squeeze(MPSNR(imageID, methodID, :)), font, "Marker", mark, "LineWidth", lineWidth);
    end
    hold off
    xlabel("\mu")
    ylabel("PSNR")
    ylim([15, 35]);
    title(imageNameList{imageID});
end
lgd = legend(methodNameModified,'interpreter','none');
lgd.Layout.Tile = 'west';
saveas(f, sprintf('%s/comparePSNR.png', outDir));
saveas(f, sprintf('%s/figs/comparePSNR.fig', outDir));
% close(f)


for imageID = 1:imageNum
    PSNRgraphs = figure("Name", imageNameList{imageID});
    PSNRgraphs.Position(3:4) = imsize_SSIM;

    tiledlayout(PSNRgraphs, 1, 1, 'TileSpacing','Compact', 'Padding', 'Compact');
    nexttile;
    hold on
    lineWidth = 0.5;
    for methodID = 1:methodNum
        switch methodID
            case 1
                font = "k:";
                mark = "o";
            case 2
                font = "r:";
                mark = "o";
            case 3
                font = "k-.";
                mark = "^";
            case 4
                font = "r-.";
                mark = "^";
            case 5
                font = "k";
                mark = "pentagram";
            case 6
                font = "r";
                mark = "pentagram";
                lineWidth = 1;
        end
        plot(range_mu, squeeze(MSSIM(imageID, methodID, :)), font, "Marker", mark, "LineWidth", lineWidth);
    end
    hold off
    xlabel("\mu")
    ylabel("SSIM")
    % ylim([0.2, 1]);
    % title(imageNameList{imageID});

    % legend(ax, methodName,'interpreter','none', 'Location', 'best')
    lgd = legend(methodNameModified,'interpreter','none');
    lgd.Layout.Tile = 'west';

    graphImageName = sprintf('%s/images/%s/SSIM.png', outDir, imageNameList{imageID});
    saveas(PSNRgraphs, graphImageName)
    graphImageName = sprintf('%s/figs/%s_SSIM.fig', outDir, imageNameList{imageID});
    saveas(PSNRgraphs, graphImageName)

    close(PSNRgraphs);
end

f = figure("Name", "SSIM");
tiledlayout(f, 1, imageNum, 'TileSpacing','Compact', 'Padding', 'Compact');
f.Position(3) = f.Position(3)*imageNum;
f.Position(3:4) = f.Position(3:4)*0.75;
for imageID = 1:imageNum
    nexttile
    hold on
    lineWidth = 0.5;
    for methodID = 1:methodNum
        switch methodID
            case 1
                font = "k:";
                mark = "o";
            case 2
                font = "r:";
                mark = "o";
            case 3
                font = "k-.";
                mark = "^";
            case 4
                font = "r-.";
                mark = "^";
            case 5
                font = "k";
                mark = "pentagram";
            case 6
                font = "r";
                mark = "pentagram";
                lineWidth = 1;
        end
        plot(range_mu, squeeze(MSSIM(imageID, methodID, :)), font, "Marker", mark, "LineWidth", lineWidth);
    end
    hold off
    xlabel("\mu")
    ylabel("SSIM")
    ylim([0, 1]);
    title(imageNameList{imageID});
end
lgd = legend(methodName,'interpreter','none');
lgd.Layout.Tile = 'west';
saveas(f, sprintf('%s/compareSSIM.png', outDir));
saveas(f, sprintf('%s/figs/compareSSIM.fig', outDir));
% close(f)

% optimized images for each mu
for muID = 1:muNum
    imageFigure = figure("Name", sprintf("Optimized images (mu=%.2f)", range_mu(muID)));
    tiledlayout(imageNum, methodNum + 2)
    for imageID = 1:imageNum
        nexttile
        imshow(u_org_buf(:,:,:,imageID))
        title("original")
    
        nexttile
        imshow(Phit(u_obsv_buf(:,:,:,imageID,shown_exNumID)));
        title("observed")
    
        for methodID = 1:methodNum
            nexttile
            imshow(u_opt_buf(:,:,:,imageID,methodID,muID,shown_exNumID));
            title(methodName{methodID},'interpreter','none')
        end
    end
    imageFigure.Position(3) = imageFigure.Position(3)*4;
    imageFigure.Position(4) = imageFigure.Position(4)*2;
    graphImageName = sprintf('%s/optimized_mu%.3f.png', outDir, range_mu(muID));
    saveas(imageFigure, graphImageName)

    if muID ~= shown_muID
        close(imageFigure);
    end
end


% Histgram of abs(Dx)
D = @(z) cat(4, z([2:size(z, 1), size(z, 1)],:,:) - z, z(:,[2:size(z, 2), size(z, 2)],:)-z);
for imageID = 1:imageNum
    Dx_org = reshape(sum(sum(abs(D(u_org_buf(:,:,:,imageID))), 4), 3), [], 1);
    Dx_opt0 = zeros(cropSize*cropSize, methodNum);
    for methodID = 1:methodNum
        Dx_opt0(:,methodID) = reshape(sum(sum(abs(D(u_opt_buf(:, :, :, imageID,methodID,shown_muID,shown_exNumID))), 4), 3), [], 1);
    end

    % cut "abs(Dx) < Dcri"
    pos = Dx_org >= Dcri;
    if sum(pos) == 0
        warning("There is no point whose abs(Dx) is bigger than %f.", Dcri);
        warning("Thus, cutting is disabled.");
        Dcri = 0;
        pos = Dx_org >= Dcri;
    end
    Dx_org = Dx_org(pos);
    Dx_opt = zeros(sum(pos), methodNum);
    for methodID = 1:methodNum
        Dx_opt(:,methodID) = Dx_opt0(pos,methodID);
    end

    % set x/y axis property
    BinLim = [floor(min(cat(2, Dx_org, Dx_opt), [], "all")), ceil(max(cat(2, Dx_org, Dx_opt), [], "all"))];
    f = figure();
    h = histogram(Dx_org, nbins, 'BinLimits', BinLim);
    ylimValue = [0, max(h.Values)*1.05];
    close(f);
    clear("h");

    for methodID = 1:methodNum
        f = figure("Name", sprintf("Histgram - %s", imageNameList{imageID}));
        f.Position(3:4) = imsize_Hist;
        tiledlayout(f, 1, 1, 'TileSpacing','Compact', 'Padding', 'tight');

        nexttile
        histogram(Dx_org, nbins, 'BinLimits', BinLim, 'FaceColor', [0.5,0.5,0.5]);
        hold on
        histogram(Dx_opt(:, methodID), nbins, 'BinLimits', BinLim);
        ylim(ylimValue);
        hold off

        saveas(f, sprintf('%s/images/%s/Histgram(Dx)_%s.png', outDir, imageNameList{imageID}, methodName{methodID}))
        saveas(f, sprintf('%s/figs/%s_%d_Histgram(Dx)_%s.fig', outDir, imageNameList{imageID}, shown_exNumID, methodName{methodID}))
        % print(f, '-depsc', sprintf('%s/figs/image%d_%d_Histgram(Dx)_%s.eps', outdir, imageID, shown_exNumID, methodName{methodID}))
        close(f)
    end
end

% all Histgram (Dx)
for imageID = 1:imageNum
    % calculate Dx
    Dx_org = reshape(sum(sum(abs(D(u_org_buf(:,:,:,imageID))), 4), 3), [], 1);
    Dx_opt0 = zeros(cropSize*cropSize, methodNum);
    for methodID = 1:methodNum
        Dx_opt0(:,methodID) = reshape(sum(sum(abs(D(u_opt_buf(:, :, :, imageID,methodID,shown_muID,shown_exNumID))), 4), 3), [], 1);
    end

    % cut "abs(Dx) < Dcri"
    pos = Dx_org >= Dcri;
    if sum(pos) == 0
        warning("There is no point whose abs(Dx) is bigger than %f.", Dcri);
        warning("Thus, cutting is disabled.");
        Dcri = 0;
        pos = Dx_org >= Dcri;
    end
    Dx_org = Dx_org(pos);
    Dx_opt = zeros(sum(pos), methodNum);
    for methodID = 1:methodNum
        Dx_opt(:,methodID) = Dx_opt0(pos,methodID);
    end

    % calculate x/y axis property
    BinLim = [floor(min(cat(2, Dx_org, Dx_opt), [], "all")), ceil(max(cat(2, Dx_org, Dx_opt), [], "all"))];
    f = figure();
    h = histogram(Dx_org, nbins, 'BinLimits', BinLim);
    ylimValue = [0, max(h.Values)*1.2];
    close(f);
    clear("h");

    f = figure("Name", sprintf("Histgram - %s", imageNameList{imageID}));
    f.Position(3:4) = 1.25*f.Position(3:4);
    tiledlayout(f, 3, 2, 'TileSpacing','Compact', 'Padding', 'Compact');
    counters = {"(a) ","(b) ","(c) ","(d) ","(e) ","(f) "};

    % DVTV/STV -----------------------------
    for methodID = 1:2:methodNum
        for i = 0:1
            nexttile
            histogram(Dx_org, nbins, 'BinLimits', BinLim, 'FaceColor', [0.5,0.5,0.5]);
            hold on
            histogram(Dx_opt(:,methodID+i), nbins, 'BinLimits', BinLim);
            ylim(ylimValue);
            xlabel(strcat(counters{methodID+i}, methodNameModified{methodID+i}), 'Interpreter','none');
            hold off
        end
    end

    lgd = legend("Ground truth", "Estimated");
    lgd.Layout.Tile = "north";
    saveas(f, sprintf('%s/%s_%d_Histgram(Dx).png', outDir, imageNameList{imageID}, shown_exNumID));
    saveas(f, sprintf('%s/figs/%s_%d_Histgram(Dx).fig', outDir, imageNameList{imageID}, shown_exNumID));
    % close(f)
end