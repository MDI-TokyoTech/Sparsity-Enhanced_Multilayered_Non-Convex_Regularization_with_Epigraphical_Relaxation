%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show and export results of DEMO
%
% input: path to all_result.mat (string)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function export_results_FRPCA(matFilePath)
%====================================================
% User settings
%====================================================
shown_exNum = 1;
shown_lambdaID = 5; % when -1, shown_lambdaID is set to best lambdaID for ER-LiGME ASNN

% Graph of PSNR/SSIM
imsize_PSNR = [360 240];
imsize_SSIM = imsize_PSNR;

% Sliced view
imsize_Slice = [270 210];

%====================================================
% Load
%====================================================
addpath code\sub_functions
if exist("matFilePath", "var")
    load(matFilePath);
end
sourceDir = para.currentDir; % input folder name
destDir = executeMfilename; % output folder name
timeStr = extractBetween(sourceDir, strlength(sourceDir)-9, strlength(sourceDir));
dayStr = extractBetween(sourceDir, strlength(sourceDir)-18, strlength(sourceDir)-11);
outDir = sprintf("output/%s/%s/%s", destDir, dayStr{1}, timeStr{1});
if ~exist(sprintf("%s/figs", outDir), "dir")
    mkdir(sprintf("%s/figs", outDir));
end
for imageID = 1:imageNum
    if ~exist(sprintf("%s/images/%s", outDir, ModifiedImageNameList{imageID}), "dir")
        mkdir(sprintf("%s/images/%s", outDir, ModifiedImageNameList{imageID}));
    end
end

if shown_lambdaID < 0
    shown_lambdaID = round(median(best_lambda_idx_PSNR(:, 4)));
end
shown_lambda1ID = ceil( shown_lambdaID / length(range_lambda2) );
shown_lambda2ID = mod(shown_lambdaID, length(range_lambda2)) + 1;
fprintf("shown results for lambda1 = %.3f, lambda2 = %.3f\n", range_lambda1(shown_lambda1ID), range_lambda2(shown_lambda2ID));


%---------------------------
% save to Excel file
%---------------------------
methodNameModified = cell(1, size(methodName, 2));
for i = 1:length(methodName)
    methodNameModified{i} = strrep(methodName{i}, '_', ' ');
    methodNameModified{i} = strrep(methodNameModified{i}, 'ER LiGME', 'ER-LiGME');
end

varTypes = cell(methodNum, 1);
for methodID = 1:methodNum
    varTypes{methodID} = 'double';
end
T = table('Size', [imageNum, methodNum], 'VariableTypes',varTypes, 'VariableNames', string(methodNameModified));
T(:, :) = num2cell(max(MPSNR, [], 3));
writetable(T, sprintf('%s/maxPSNR.xlsx', currentDir));

T = table('Size', [imageNum, methodNum], 'VariableTypes',varTypes, 'VariableNames', string(methodNameModified));
T(:, :) = num2cell(max(MSSIM, [], 3));
writetable(T, sprintf('%s/maxSSIM.xlsx', currentDir));

% only image 1
varTypes = cell(length(range_lambda1) + 1, 1);
varTypes{1} = 'string';
for lambda1ID = 1:length(range_lambda1)
    varTypes{lambda1ID+1} = 'double';
end
for lambda2ID = 1:length(range_lambda2)
    T = table('Size', [methodNum, 1 + length(range_lambda1)], 'VariableTypes',varTypes, 'VariableNames', ['Method', string(range_lambda1)]);
    T(:, 1) = methodNameModified';
    for methodID = 1:methodNum
        lambdaIdx = (lambda2ID - 1)*length(range_lambda2);
        lambdaIdx = lambdaIdx+1:lambdaIdx+length(range_lambda1);
        T(methodID, 2:end) = num2cell(squeeze(MPSNR(1,methodID,lambdaIdx))');
    end
    writetable(T, sprintf('%s/%s_PSNR(lambda2_%.3f).xlsx', currentDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID)));
end
for lambda2ID = 1:length(range_lambda2)
    T = table('Size', [methodNum, 1 + length(range_lambda1)], 'VariableTypes',varTypes, 'VariableNames', ['Method', string(range_lambda1)]);
    T(:, 1) = methodNameModified';
    for methodID = 1:methodNum
        lambdaIdx = (lambda2ID - 1)*length(range_lambda2);
        lambdaIdx = lambdaIdx+1:lambdaIdx+length(range_lambda1);
        T(methodID, 2:end) = num2cell(squeeze(MSSIM(1,methodID,lambdaIdx))');
    end
    writetable(T, sprintf('%s/%s_SSIM(lambda2_%.3f).xlsx', currentDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID)));
end


%---------------------------
% Difference between enhanced and not enhanced results
%---------------------------
diary(sprintf("%s/difference.txt", outDir));
for imageID = 1:imageNum
for lambdaID = 1:length(range_lambda2):lambdaNum
for l2ID = 0:length(range_lambda2)-1
for pairID = 1:methodPairNum
    pair = methodPair{pairID};
    fprintf('%s(lambda1=%.3f,lambda2=%.3f)---difference_of_method%d,%d(L2norm): %f\n', ...
    ModifiedImageNameList{imageID}, range_lambda1(ceil(lambdaID/length(range_lambda2))), range_lambda2(l2ID+1), pair(1), pair(2), Mdiff_opt_L(imageID, lambdaID+l2ID));
end
end
end
end
for imageID = 1:imageNum
for l1ID = 1:length(range_lambda1)
for l2ID = 1:length(range_lambda2)
for pairID = 1:methodPairNum
    pair = methodPair{pairID};
    for i = 1:2
        NNs = zeros(exNumber_max, 1);
        for exNumber = 1:exNumber_max
            u_opt = im2double(imread(sprintf('%s/images/%s/%02d/%s/lambda1_%.3f_lambda2_%.3f.%s', ...
                currentDir, ModifiedImageNameList{imageID}, exNumber, methodName{pair(i)}, ...
                range_lambda1( l1ID ),...
                range_lambda2( l2ID ),...
                imageFormat)));
            [~,S,~] = svd(u_opt);
            NNs(exNumber) = sum(S, 'all');
        end
        fprintf('%s(lambda1=%.3f,lambda2=%.3f)---NN(method%d-%s): %.15f\n', ...
        ModifiedImageNameList{imageID}, range_lambda1(l1ID), range_lambda2(l2ID), pair(i), methodName{pair(i)}, mean(NNs));
    end
end
end
end
end
diary off


%====================================================
% Save figures
%====================================================
%---------------------------
% PSNR/SSIM regarding changing lambda1
%---------------------------
for imageID = 1:imageNum
    for lambda2ID = 1:length(range_lambda2)
        lambdaID = lambda2ID:length(range_lambda2):lambda2ID+length(range_lambda2)*length(range_lambda1)-1;
    
        PSNRgraphs = figure("Name", ModifiedImageNameList{imageID});
        PSNRgraphs.Position(3:4) = imsize_PSNR;
        tiledlayout(PSNRgraphs, 1, 1, 'TileSpacing','Compact', 'Padding', 'Compact');
        nexttile;
        hold on
        for methodID = 1:4
            switch methodID
                case 1
                    font = "b:";
                    mark = "o";
                case 2
                    font = "b";
                    mark = "o";
                case 3
                    font = "r:";
                    mark = "pentagram";
                case 4
                    font = "r";
                    mark = "pentagram";
            end
            plot(range_lambda1, squeeze(MPSNR(imageID, methodID, lambdaID)), font, "Marker", mark);
        end
        hold off
        xlabel("\lambda_1")
        ylabel("PSNR")
        % lgd = legend(methodNameModified);
        lgd = legend("NN", "L-NN", "ASNN", "EL-ASNN");
        lgd.Layout.Tile = 'west';
        xlim([0, max(range_lambda1, [], "all")])
    
        graphImageName = sprintf('%s/images/%s/PSNR(lambda2_%.3f).png', outDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID));
        saveas(PSNRgraphs, graphImageName)
        graphImageName = sprintf('%s/figs/%s_PSNR(lambda2_%.3f).fig', outDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID));
        saveas(PSNRgraphs, graphImageName)
        % graphImageName = sprintf('%s/images/%s_PSNR(lambda2_%.3f).eps', outDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID));
        % saveas(PSNRgraphs, graphImageName, 'epsc')

        if mod(shown_lambdaID, length(range_lambda2)) + 1 ~= lambda2ID
            close(PSNRgraphs)
        end
    end
end
for imageID = 1:imageNum
    for lambda2ID = 1:length(range_lambda2)
        lambdaID = lambda2ID:length(range_lambda2):lambda2ID+length(range_lambda2)*length(range_lambda1)-1;
    
        PSNRgraphs = figure("Name", ModifiedImageNameList{imageID});
        PSNRgraphs.Position(3:4) = imsize_SSIM;
        tiledlayout(PSNRgraphs, 1, 1, 'TileSpacing','Compact', 'Padding', 'Compact');
        nexttile;
        hold on
        for methodID = 1:4
            switch methodID
                case 1
                    font = "b:";
                    mark = "o";
                case 2
                    font = "b";
                    mark = "o";
                case 3
                    font = "r:";
                    mark = "pentagram";
                case 4
                    font = "r";
                    mark = "pentagram";
            end
            plot(range_lambda1, squeeze(MSSIM(imageID, methodID, lambdaID)), font, "Marker", mark);
        end
        hold off
        xlabel("\lambda_1")
        ylabel("SSIM")
        % lgd = legend(methodNameModified);
        lgd = legend("NN", "L-NN", "ASNN", "EL-ASNN");
        lgd.Layout.Tile = 'west';
        xlim([0, max(range_lambda1, [], "all")])
        ylim([0,1]);
    
        graphImageName = sprintf('%s/images/%s/SSIM(lambda2_%.3f).png', outDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID));
        saveas(PSNRgraphs, graphImageName)
        graphImageName = sprintf('%s/figs/%s_SSIM(lambda2_%.3f).fig', outDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID));
        saveas(PSNRgraphs, graphImageName)
        % graphImageName = sprintf('%s/images/%s_PSNR(lambda2_%.3f).eps', outDir, ModifiedImageNameList{imageID}, range_lambda2(lambda2ID));
        % saveas(PSNRgraphs, graphImageName, 'epsc')

        if mod(shown_lambdaID, length(range_lambda2)) + 1 ~= lambda2ID
            close(PSNRgraphs)
        end
    end
end

%--------------------
% optimized signals for each lambda1
%--------------------
for lambda1ID = 1:length(range_lambda1)
    imageFigure = figure();
    tiledlayout(imageNum*length(range_lambda2), methodNum+2, 'TileSpacing','Compact', 'Padding', 'tight')
    
    for imageID = 1:imageNum
        for lambda2ID = 1:length(range_lambda2)    
            nexttile;
            imageName = ModifiedImageNameList{imageID};
            u_org = im2double(imread(sprintf('%s/images/%s/org.%s', sourceDir, imageName, imageFormat)));
            meshz(u_org)
            zlim([-0.15, 1.15])
            title("original")
        
            nexttile;
            u_obsv = im2double(imread(sprintf('%s/images/%s/%02d/obsv.%s', sourceDir, imageName, shown_exNum, imageFormat)));
            meshz(u_obsv)
            zlim([-0.15, 1.15])
            title("observed")
        
            for methodID = 1:methodNum
                nexttile;
                u_opt = im2double(imread(sprintf('%s/images/%s/%02d/%s/lambda1_%.3f_lambda2_%.3f.%s', ...
                    sourceDir, imageName, shown_exNum, methodName{methodID}, ...
                    range_lambda1( lambda1ID ),...
                    range_lambda2( lambda2ID ),...
                    imageFormat)));
                meshz(u_opt)
                zlim([-0.15, 1.15])
                title(sprintf("%s\nλ1=%.2f, λ2=%.2f", methodName{methodID}, range_lambda1( lambda1ID ), range_lambda2( lambda2ID )),'interpreter','none')
            end
        end
    end
    imageFigure.Position(3) = imageFigure.Position(3)*1.5;
    imageFigure.Position(4) = imageFigure.Position(4)*0.375;
    graphImageName = sprintf('%s/optimized_lambda1_%.3f.png', outDir, range_lambda1( lambda1ID ));
    saveas(imageFigure, graphImageName)
    graphImageName = sprintf('%s/figs/optimized_lambda1_%.3f.fig', outDir, range_lambda1( lambda1ID ));
    saveas(imageFigure, graphImageName)
    % graphImageName = sprintf('%s/optimized_lambda1_%.3f.eps', outDir, range_lambda1( lambda1ID ));
    % saveas(imageFigure, graphImageName, 'epsc')
    
    if floor(shown_lambdaID / length(range_lambda2)) ~= lambda1ID
        close(imageFigure);
    end
end

%--------------------
% signals in 3D view
%--------------------
for imageID = 1:imageNum
    imageName = ModifiedImageNameList{imageID};

    imageFigure = figure();
    imsize = imageFigure.Position(3:4);
    imsize(1) = imsize(1)*0.3;
    imsize(2) = imsize(2)*0.375;

    imageFigure.Position(3:4) = imsize;
    tiledlayout(1, 1, 'TileSpacing','Compact', 'Padding', 'tight')
    nexttile;
    u_org = im2double(imread(sprintf('%s/images/%s/org.%s', sourceDir, imageName, imageFormat)));
    meshz(u_org)
    zlim([-0.15, 1.15])
    graphImageName = sprintf('%s/images/%s/3Dview_org.png', outDir, imageName);
    saveas(imageFigure, graphImageName)
    graphImageName = sprintf('%s/figs/%s_3Dview_org.fig', outDir, imageName);
    saveas(imageFigure, graphImageName)
    % graphImageName = sprintf('%s/images/%s/3Dview_org.eps', outDir, imageName);
    % saveas(imageFigure, graphImageName, 'epsc')
    close(imageFigure);

    imageFigure = figure();
    imageFigure.Position(3:4) = imsize;
    tiledlayout(1, 1, 'TileSpacing','Compact', 'Padding', 'tight')
    nexttile;
    u_obsv = im2double(imread(sprintf('%s/images/%s/%02d/obsv.%s', sourceDir, imageName, shown_exNum, imageFormat)));
    meshz(u_obsv)
    zlim([-0.15, 1.15])
    graphImageName = sprintf('%s/images/%s/%02d/3Dview_obsv.png', outDir, imageName, shown_exNum);
    saveas(imageFigure, graphImageName)
    graphImageName = sprintf('%s/figs/%s_%02d_3Dview_obsv.fig', outDir, imageName, shown_exNum);
    saveas(imageFigure, graphImageName)
    % graphImageName = sprintf('%s/images/%s/%02d/3Dview_obsv.eps', outDir, imageName, shown_exNum);
    % saveas(imageFigure, graphImageName, 'epsc')
    close(imageFigure);

    for methodID = 1:methodNum
        imageFigure = figure();
        imageFigure.Position(3:4) = imsize;
        tiledlayout(1, 1, 'TileSpacing','Compact', 'Padding', 'tight')

        lambda1ID = ceil( shown_lambdaID / length(range_lambda2) );
        lambda2ID = mod(shown_lambdaID, length(range_lambda2)) + 1;

        nexttile;
        u_opt = im2double(imread(sprintf('%s/images/%s/%02d/%s/lambda1_%.3f_lambda2_%.3f.%s', ...
            sourceDir, imageName, shown_exNum, methodName{methodID}, ...
            range_lambda1( lambda1ID ),...
            range_lambda2( lambda2ID ),...
            imageFormat)));
        meshz(u_opt)
        zlim([-0.15, 1.15])

        graphImageName = sprintf('%s/images/%s/%02d/3Dview(lambda1_%.3f_lambda2_%.3f)_%s.png', outDir, imageName, shown_exNum, range_lambda1( lambda1ID ), range_lambda2( lambda2ID ), methodName{methodID});
        saveas(imageFigure, graphImageName)
        graphImageName = sprintf('%s/figs/%s_%02d_3Dview(lambda1_%.3f_lambda2_%.3f)_%s.fig', outDir, imageName, shown_exNum, range_lambda1( lambda1ID ), range_lambda2( lambda2ID ), methodName{methodID});
        saveas(imageFigure, graphImageName)
        % graphImageName = sprintf('%s/images/3Dview_%s.eps', outDir, methodName{methodID});
        % saveas(imageFigure, graphImageName, 'epsc')
        close(imageFigure);
    end
end


%---------------------------
% Sliced view
%---------------------------
for imageID = 1:imageNum
    imageName = ModifiedImageNameList{imageID};
    u_org = im2double(imread(sprintf('%s/images/%s/org.%s', sourceDir, imageName, imageFormat)));

    [rows,cols,dim] = size(u_org);
    u_opt = zeros(rows,cols,dim,methodNum);

    for lambda2ID = 1:length(range_lambda2)
        for lambda1ID = 1:length(range_lambda1)
            f = figure("Name", sprintf("%s", imageName));
            tiledlayout(f, 1, 2, 'TileSpacing','Compact', 'Padding', 'tight');

            for methodID = 1:methodNum
                load(sprintf("%s/mat/%s/%02d/%s/theta%.3f_lambda1_%.3f_lambda2_%.3f.mat", sourceDir,...
                    imageName, shown_exNum, methodName{methodID},...
                    thetaList_1(methodID), range_lambda1(lambda1ID), ...
                    range_lambda2(lambda2ID)), "x");
                u_opt(:,:,:,methodID) = gather(x{1});
            end

            nexttile
            plot(1:rows, u_org(:, round(cols/2), 1), "k")
            hold on
            plot(1:rows, u_opt(:, round(cols/2), 1, 1), "b:", "LineWidth", 1)
            plot(1:rows, u_opt(:, round(cols/2), 1, 2), "b", "LineWidth", 1)
            hold off
            ylim([0, 1.1]);
            lgd = legend("GT", "NN", "L-NN", 'Location', 'northwest');
            lgd.Layout.Tile = 'north';

            nexttile
            plot(1:rows, u_org(:, round(cols/2), 1), "k")
            hold on
            plot(1:rows, u_opt(:, round(cols/2), 1, 3), "b:", "LineWidth", 1)
            plot(1:rows, u_opt(:, round(cols/2), 1, 4), "r", "LineWidth", 1)
            hold off
            ylim([0, 1.1]);
            lgd = legend("GT", "ASNN", "EL"+ newline  +"-ASNN", 'Location', 'northwest');
            lgd.Layout.Tile = 'north';

            f.Position(3:4) = imsize_Slice;
            saveas(f, sprintf('%s/images/%s/SlicedView_halfCol(lambda1_%.3f_lambda2_%.3f).png', outDir, imageName, range_lambda1(lambda1ID), range_lambda2(lambda2ID)))
            saveas(f, sprintf('%s/figs/%s_SlicedView_halfCol(lambda1_%.3f_lambda2_%.3f).fig', outDir, imageName, range_lambda1(lambda1ID), range_lambda2(lambda2ID)))
            % saveas(f, sprintf('%s/images/%s/SlicedView_halfCol(lambda1_%.3f_lambda2_%.3f).eps', outDir, imageName, range_lambda1(lambda1ID), range_lambda2(lambda2ID)), 'epsc')

            if shown_lambdaID ~= (lambda1ID - 1)*length(range_lambda2) + lambda2ID
                close(f)
            end
        end
    end
end
