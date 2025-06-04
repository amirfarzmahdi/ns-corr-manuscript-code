% description: noise correlation for small and large natural images, Ensembles data 150Anes
% author: Amirhossein Farzmahdi
% last update: May 15th 2024

clear
close all
clc

% add path
addpath(genpath('functions/'))

% load Sessions
load('data/Sessions_Natural_150.mat')

% settings
stim_blk_ratio = (diff(Session(1).stimwindow) + 1) / (diff(Session(1).blkwindow) + 1);
Sessions = CatStructFields(Session(1:4), 1);

centered_idx = sqrt(sum(Sessions.XYch.^2,2)) <= 25;
offcentered_idx = sqrt(sum(Sessions.XYch.^2,2)) >= 30 &...
                  sqrt(sum(Sessions.XYch.^2,2)) <= 60; % close to center probes
              
centered_tril_mat = tril(ones(sum(centered_idx)),-1);
centered_tril_mat(centered_tril_mat==0) = NaN;

CC_small = NaN(Sessions.NIm(1),sum(Sessions.N),sum(Sessions.N));
CC_large = NaN(Sessions.NIm(1),sum(Sessions.N),sum(Sessions.N));

CC_small_centered = [];
CC_large_centered = [];
CC_small_mixed = [];
CC_large_mixed = [];

MM_small_centered = [];
MM_large_centered = [];
MM_small_mixed = [];
MM_large_mixed = [];

count = 0;
for i_image = 1:Sessions.NIm(1)
    count = count + 1;
    
    %  responsive cells per image
    thr1 = max(Sessions.spontaneous,0.1);
    thr2 = max((Sessions.spontaneous + 0.1 * Sessions.stdspontaneous),0.1);
    
    resp_idx = zeros(sum(Sessions.N),1);
    resp_idx(centered_idx,1)  = (Sessions.MM_small(centered_idx,i_image) >= thr2(centered_idx));
    resp_idx(offcentered_idx,1)  = (Sessions.MM_small(offcentered_idx,i_image) <= thr1(offcentered_idx)) & (Sessions.MM_large(offcentered_idx,i_image) >= thr2(offcentered_idx));
    
    tmp = squeeze(Sessions.resp_small(:,i_image,:))';
    tmp(:,~resp_idx) = NaN;
    CC_small(count,:,:) = corr(tmp,'rows','all');
    
    tmp = squeeze(Sessions.resp_large(:,i_image,:))';
    tmp(:,~resp_idx) = NaN;
    CC_large(count,:,:) = corr(tmp,'rows','all');
    
    % centered
    tmpScent = squeeze(CC_small(count,centered_idx,centered_idx)) .* centered_tril_mat;
    tmpLcent = squeeze(CC_large(count,centered_idx,centered_idx)) .* centered_tril_mat;
    CC_small_centered = [CC_small_centered; (tmpScent(isfinite(tmpScent(:) .* tmpLcent(:))))];
    CC_large_centered = [CC_large_centered; (tmpLcent(isfinite(tmpScent(:) .* tmpLcent(:))))];
    
    % mixed
    tmpSmixed = squeeze(CC_small(count,centered_idx,offcentered_idx));
    tmpLmixed = squeeze(CC_large(count,centered_idx,offcentered_idx));
    CC_small_mixed = [CC_small_mixed; (tmpSmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
    CC_large_mixed = [CC_large_mixed; (tmpLmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];

    sNeurons_cent = find(centered_idx);
    sNeurons_offcent = find(offcentered_idx);
    
    % mixed
    [cent_idx, offcent_idx] = find(isfinite(tmpSmixed .* tmpLmixed));
    mean_pair_small = mean([Sessions.MM_small(sNeurons_offcent(offcent_idx),count),...
        Sessions.MM_small(sNeurons_cent(cent_idx),count)],2);
    mean_pair_large = mean([Sessions.MM_large(sNeurons_offcent(offcent_idx),count),...
        Sessions.MM_large(sNeurons_cent(cent_idx),count)],2);
    
    MM_small_mixed = [MM_small_mixed; mean_pair_small];
    MM_large_mixed = [MM_large_mixed; mean_pair_large];
    
    % centered
    [cent_idx1, cent_idx2] = find(isfinite(tmpScent .* tmpLcent));
    mean_pair_small = mean([Sessions.MM_small(sNeurons_cent(cent_idx1),count),...
        Sessions.MM_small(sNeurons_cent(cent_idx2),count)],2);
    mean_pair_large = mean([Sessions.MM_large(sNeurons_cent(cent_idx1),count),...
        Sessions.MM_large(sNeurons_cent(cent_idx2),count)],2);
    
    MM_small_centered = [MM_small_centered; mean_pair_small];
    MM_large_centered = [MM_large_centered; mean_pair_large];
    
end

% save variables
num_neurons(1,1) = sum(centered_idx);
num_neurons(1,2) = sum(offcentered_idx);

save rsc_anes150...
    CC_small_centered...
    CC_large_centered...
    CC_small_mixed...
    CC_large_mixed...
    MM_small_centered...
    MM_large_centered...
    MM_small_mixed...
    MM_large_mixed...
    num_neurons