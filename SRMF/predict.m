% Original code: https://github.com/linwang1982/SRMF
% Code edited to run SRMF on our data

clear;clc;

% Set the directory where the seedi folders are located
dir_path = '/Users/melodyparker/Documents/DRP/SRMF/SRMF_datasets';

% Extract all data set directories to loop through
datasets = strsplit(genpath(dir_path), pathsep());
datasets = datasets(contains(datasets, 'seed'));

% Loop through data sets
for i = 1:numel(datasets)
    % set current directory
    curr_dir = datasets{i};
    addpath(curr_dir);
    % extract data
    resp_mat = dir(fullfile(curr_dir, '*resp.mat'));
    drugsim_fig_mt_mat = dir(fullfile(curr_dir, '*Drugsim_fig_mt.mat'));
    cellsim_probe_mat = dir(fullfile(curr_dir, '*Cellsim_probe.mat'));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Start code
    load(resp_mat.name);
    % use drug as row indix and cell line as column index
    resp = resp';
    scale1 = resp(~isnan(resp));
    num = resp./max(max(scale1),abs(min(scale1)));
    load(drugsim_fig_mt_mat.name);
    load(cellsim_probe_mat.name);
    
    drugwisecorr = NaN(size(num,1),1);
    drugwise_qt = NaN(size(num,1),1);
    drugwiseerr = NaN(size(num,1),1);
    drugwiseerr_qt = NaN(size(num,1),1);
    drugwiserepn = NaN(size(num,1),1);
    
    i1 = -2; i3 = -2;
    % K = 45; lambda_l = 2^i1; lambda_d = 2^i2; lambda_c = 2^i3; max_iter=50; seed=50;
    K = 45; lambda_l = 2^i1; lambda_d = 0; lambda_c = 2^i3; max_iter=50; seed=50;
    curnum = num;
    W = ~isnan(curnum);
    curnum(isnan(curnum)) = 0;
    [U,V] = CMF(W,curnum,Drugsim_fig_mt,Cellsim_probe,lambda_l,lambda_d,lambda_c,K,max_iter,seed);
    num = num *max(max(scale1),abs(min(scale1)));
    numpred = U*V'*max(max(scale1),abs(min(scale1)));
    for d = 1:size(num,1)
        curtemp1 = num(d,:);
        y1 = prctile(curtemp1,75);
        xia1 = find(curtemp1 >= y1);
        y2 = prctile(curtemp1,25);
        xia2 = find(curtemp1 <= y2);
        xia = [xia1,xia2];
        drugwise_qt(d) = corr(curtemp1(xia)',numpred(d,xia)');
        drugwiseerr_qt(d) = sqrt(sum((curtemp1(xia)-numpred(d,xia)).^2)/sum(~isnan(curtemp1(xia))));
        curtemp2 = numpred(d,:);
        curtemp2(isnan(curtemp1)) = [];
        curtemp1(isnan(curtemp1)) = [];
        drugwiserepn(d) = length(curtemp1);  
        drugwisecorr(d) = corr(curtemp1',curtemp2');
        drugwiseerr(d) = sqrt(sum((curtemp1-curtemp2).^2)/sum(~isnan(curtemp1)));
    end
    % save('drugwise_predict1.mat');
    out_prefix = strrep(resp_mat.name, 'resp.mat', '');
    % save(fullfile(curr_dir, 'drugwise_predict1.mat'));
    save(fullfile(curr_dir, [out_prefix, 'drugwise_predict1.mat']));
    % load(fullfile(curr_dir, [out_prefix, 'drugwise_predict1.mat']))

    % Write response matrix to csv
    F = U*transpose(V);
    G = transpose(F);
    writematrix(G, fullfile(curr_dir, [out_prefix, 'G.csv']));

    % Remove path
    rmpath(curr_dir);
end
