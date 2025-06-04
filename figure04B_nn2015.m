% description: noise correlation for small and large natural images, NN
% 2015 data, centered and mixed
% author: Amirhossein Farzmahdi
% last update: May 17th 2024

clear
close all
clc

% data directory
data_dir = '/data/NN2015/';

% add path
addpath(genpath('/functions/'))

% settings
nsession = 7; % number of sessions
reps = 20; % number of repeats
latency = 50;
stimwindow = [1 105];
blkwindow = [161 210];

stim_blk_ratio =  (diff(stimwindow) +1) / (diff(blkwindow) + 1);

CC_small_centered = cell(1, nsession);
CC_large_centered = cell(1, nsession);
CC_small_mixed = cell(1, nsession);
CC_large_mixed = cell(1, nsession);

MM_small_centered = cell(1, nsession);
MM_large_centered = cell(1, nsession);
MM_small_mixed = cell(1, nsession);
MM_large_mixed = cell(1, nsession);

for i_session = 1:nsession
    
    % load session & pre-processing
    load([data_dir '0' num2str(i_session) '.mat'])
    
    % shift train window for response latency
    resp_train = resp_train(:,:,:,1:end-1);
    resp_train_blk = resp_train_blk(:,:,:,1:end-1);
    tmp = resp_train; % [#neurons #stimuli #repeats #millisecond]
    tmp(:,:,:,1:end-latency) = resp_train(:,:,:,latency+1:end);
    tmp(:,:,:,end-latency+1:end) = resp_train_blk(:,:,:,1:latency);
    resp_train = tmp;
    
    % small and large spike count
    resp = sum(resp_train,4,'omitnan'); % [#neurons #stimuli #repeats]
    tmp = squeeze(resp(:,1:(2 * 9 * 30),:));
    
    tmp = reshape(tmp,size(tmp,1),2,9,30,reps);
    resp_small =  squeeze(tmp(:,1,:,:,:));
    resp_large =  squeeze(tmp(:,2,:,:,:));
    
    resp_small = reshape(resp_small,[size(resp_small,1),size(resp_small,2)*size(resp_small,3),size(resp_small,4)]);
    resp_large = reshape(resp_large,[size(resp_large,1),size(resp_large,2)*size(resp_large,3),size(resp_large,4)]);
    
    MM_small = mean(resp_small,3,'omitnan'); % small natural images
    MM_large = mean(resp_large,3,'omitnan'); % large natural images
   
    % spontaneous response
    tmp = nansum(resp_train_blk(:,1:2*9*30,:,blkwindow(1):blkwindow(2)),4);
    spontaneous = mean(tmp(:,:),2,'omitnan');
    stdspontaneous = std(tmp(:,:),[],2,'omitnan');
    
    % surround suppression
    NIm = size(resp_large,2);
    N = size(resp_large,1);
    T = size(resp_large,3);
    
    % centered index
    centered_idx = sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 25;
    
    offcentered_idx = ~isnan(RF_SPATIAL(:,1)) &...
        sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) >= 30 &...
        sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 150;
        
    centered_tril_mat = tril(ones(sum(centered_idx)),-1);
    centered_tril_mat(centered_tril_mat==0) = NaN;

    % rsc sorted by firing rate and distance
    CC_small = NaN(NIm,N,N);
    CC_large = NaN(NIm,N,N);
    
    count = 0;
    for i_image = 1:NIm
        count = count + 1;
        
        %  responsive cells per image
        thr1 = spontaneous;
        thr2 = max((spontaneous + 1 * stdspontaneous) * stim_blk_ratio ,0.1);
        
        resp_idx = zeros(N(1),1);
        resp_idx(centered_idx,1)  = (MM_small(centered_idx,i_image) >= thr2(centered_idx));
        resp_idx(offcentered_idx,1)  = (MM_small(offcentered_idx,i_image) <= thr1(offcentered_idx)) & (MM_large(offcentered_idx,i_image) >= thr2(offcentered_idx));
        
        tmp = squeeze(resp_small(:,i_image,:))';
        tmp(:,~resp_idx) = NaN;
        CC_small(count,:,:) = corr(tmp,'rows','all');
        
        tmp = squeeze(resp_large(:,i_image,:))';
        tmp(:,~resp_idx) = NaN;
        CC_large(count,:,:) = corr(tmp,'rows','all');
        
        % centered
        tmpScent = squeeze(CC_small(count,centered_idx,centered_idx)) .* centered_tril_mat;
        tmpLcent = squeeze(CC_large(count,centered_idx,centered_idx)) .* centered_tril_mat;
        CC_small_centered{1, i_session} = [CC_small_centered{1, i_session}; (tmpScent(isfinite(tmpScent(:) .* tmpLcent(:))))];
        CC_large_centered{1, i_session} = [CC_large_centered{1, i_session}; (tmpLcent(isfinite(tmpScent(:) .* tmpLcent(:))))];
        
        % mixed
        tmpSmixed = squeeze(CC_small(count,centered_idx,offcentered_idx));
        tmpLmixed = squeeze(CC_large(count,centered_idx,offcentered_idx));
        CC_small_mixed{1, i_session} = [CC_small_mixed{1, i_session}; (tmpSmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
        CC_large_mixed{1, i_session} = [CC_large_mixed{1, i_session}; (tmpLmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
        
        sNeurons_cent = find(centered_idx);
        sNeurons_offcent = find(offcentered_idx);
        
        % mixed
        [cent_idx, offcent_idx] = find(isfinite(tmpSmixed .* tmpLmixed));
        mean_pair_small = mean([MM_small(sNeurons_offcent(offcent_idx),count),...
            MM_small(sNeurons_cent(cent_idx),count)],2);
        mean_pair_large = mean([MM_large(sNeurons_offcent(offcent_idx),count),...
            MM_large(sNeurons_cent(cent_idx),count)],2);
        
        MM_small_mixed{1, i_session} = [MM_small_mixed{1, i_session}; mean_pair_small];
        MM_large_mixed{1, i_session} = [MM_large_mixed{1, i_session}; mean_pair_large];
        
        % centered
        [cent_idx1, cent_idx2] = find(isfinite(tmpScent .* tmpLcent));
        mean_pair_small = mean([MM_small(sNeurons_cent(cent_idx1),count),...
            MM_small(sNeurons_cent(cent_idx2),count)],2);
        mean_pair_large = mean([MM_large(sNeurons_cent(cent_idx1),count),...
            MM_large(sNeurons_cent(cent_idx2),count)],2);
        
        MM_small_centered{1, i_session} = [MM_small_centered{1, i_session}; mean_pair_small];
        MM_large_centered{1, i_session} = [MM_large_centered{1, i_session}; mean_pair_large];
        
    end
    
    num_neurons(i_session,1) = sum(centered_idx);
    num_neurons(i_session,2) = sum(offcentered_idx);
end

% save variables
save rsc_sessions_NN2015 ...
        CC_small_centered...
        CC_large_centered...
        CC_small_mixed...
        CC_large_mixed...
        MM_small_centered...
        MM_large_centered...
        MM_small_mixed...
        MM_large_mixed...
        num_neurons
    