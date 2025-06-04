% figure 6C: drsc across distance and rsignal
% author: Amirhossein Farzmahdi
% last update: May 20th 2024

clear
close all
clc

% data directory
data_dir = '/data/NN2015/';

% add path
addpath(genpath('/functions/'))

% settings
nsession = 9; % 7 sessions NN2015, 1 session Anes136
nbins = 20;
min_entry_thr = 15; % set min number of entries to remove noises on 2D map
nlocs = 20; % number of distance bins
nrignals = 15; % number of risginal bins

SS_all = [];
DD_all = [];
CC_small_all = [];
CC_large_all = [];

for i_session = 1:nsession % 1:7 nn2015, 8: anes136, 9: anes150
    
    SS_session = [];
    DD_session = [];
    CC_small_session = [];
    CC_large_session = [];
    
    if ismember(i_session, [1,2,3,4,5,6,7])  % 1:7 nn2015
        
        % settings
        stimwindow = [1 105];
        blkwindow = [161 210];
        reps = 20;
        latency = 50;
        
        stim_blk_ratio =  (diff(stimwindow) + 1)/(diff(blkwindow) + 1);
        
        % load session & pre-processing
        load([data_dir '0' num2str(i_session) '.mat'])
        
        % shift train window for response latency
        resp_train = resp_train(:,:,:,1:end-1);
        resp_train_blk = resp_train_blk(:,:,:,1:end-1);
        tmp = resp_train; % [#neurons #stimuli #repeats #millisecond]
        tmp(:,:,:,1:end-latency) = resp_train(:,:,:,latency+1:end);
        tmp(:,:,:,end-latency+1:end) = resp_train_blk(:,:,:,1:latency);
        resp_train = tmp;
        
        % select neurons based on RF position
        centered_idx = ~isnan(RF_SPATIAL(:,1)) &...
            sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 25;
        
        offcentered_idx = ~isnan(RF_SPATIAL(:,1)) &...
            sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) >= 30 &...
            sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 150;
        
        mixed_idx = [find(centered_idx); find(offcentered_idx)];
        
        % generate an upper triangular mask including the diagonal for the
        % centered pairs
        mask = triu(true(sum(centered_idx), sum(centered_idx)));
        
        display(['centered: ' num2str(sum(centered_idx)) '   ' 'offcentered: ' num2str(sum(offcentered_idx))])
        
        % small and large spike count
        resp = sum(resp_train,4,'omitnan'); % [#neurons #stimuli #repeats]
        tmp = squeeze(resp(:,1:(2*9*30),:));
        
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
        
        NIm = size(MM_large,2);
        N = size(MM_large,1);
        
        % rsignal
        rsignal = corr(MM_large','rows','all');
        rsignal_centered_versus_mixed = rsignal(centered_idx,:);
        rsignal_centered_versus_mixed = rsignal_centered_versus_mixed(:,mixed_idx);
        
        % pdist
        dists = pdist([RF_SPATIAL(:,1),RF_SPATIAL(:,2)],'euclidean');
        dists = squareform(dists);
        dist_centered_versus_mixed = dists(centered_idx,:);
        dist_centered_versus_mixed = dist_centered_versus_mixed(:,mixed_idx);
        
        % replace the upper triangle of centered pairs with nan
        rsignal_centered_versus_mixed(mask) = NaN;
        dist_centered_versus_mixed(mask) = NaN;
        
    elseif i_session == 8 % anes 136
        % load data
        load('/data/Sessions_NaturalEnsemble_136.mat')
        
        stim_blk_ratio = (diff(Session.stimwindow) + 1) / (diff(Session.blkwindow) + 1);
        
        spontaneous = Session.spontaneous;
        stdspontaneous = Session.stdspontaneous;
        
        NIm = size(Session.MM_large,2);
        N = size(Session.MM_large,1);
        
        MM_small = Session.MM_small;
        MM_large = Session.MM_large;
        resp_small = Session.resp_small;
        resp_large = Session.resp_large;
        
        % select neurons based on RF position
        centered_idx = sqrt(sum(Session.XYch.^2,2)) <= 25;
        
        offcentered_idx = sqrt(sum(Session.XYch.^2,2)) >= 30 & ...
            sqrt(sum(Session.XYch.^2,2)) <= 150;
        
        display(['centered: ' num2str(sum(centered_idx)) '   ' 'offcentered: ' num2str(sum(offcentered_idx))])
        
        mixed_idx = [find(centered_idx); find(offcentered_idx)];
        
        % generate an upper triangular mask including the diagonal for the
        % centered pairs
        mask = triu(true(sum(centered_idx), sum(centered_idx)));
        
        % rsignal
        rsignal = corr(Session.MM_large','rows','all');
        rsignal_centered_versus_mixed = rsignal(centered_idx,:);
        rsignal_centered_versus_mixed = rsignal_centered_versus_mixed(:,mixed_idx);
        
        % pdist
        dists = pdist([Session.XYch(:,1),Session.XYch(:,2)],'euclidean');
        dists = squareform(dists);
        dist_centered_versus_mixed = dists(centered_idx,:);
        dist_centered_versus_mixed = dist_centered_versus_mixed(:,mixed_idx);
        
        % replace the upper triangle of centered pairs with nan
        rsignal_centered_versus_mixed(mask) = NaN;
        dist_centered_versus_mixed(mask) = NaN;
        
    elseif i_session == 9 % anes150
        % load data
        load('data/anes150/Sessions_Natural_150.mat')
        
        stim_blk_ratio = (diff(Session(1).stimwindow) + 1) / (diff(Session(1).blkwindow) + 1);
        Sessions = CatStructFields(Session(1:4), 1);
        
        centered_idx = sqrt(sum(Sessions.XYch.^2,2)) <= 25;
        offcentered_idx = sqrt(sum(Sessions.XYch.^2,2)) >= 30 &...
            sqrt(sum(Sessions.XYch.^2,2)) <= 60;
        
        disp(['centered ',num2str(sum(centered_idx)), '   offcentered ',num2str(sum(offcentered_idx))])
        
        spontaneous = Sessions.spontaneous;
        stdspontaneous = Sessions.stdspontaneous;
        
        NIm = size(Sessions.MM_large,2);
        N = size(Sessions.MM_large,1);
        
        MM_small = Sessions.MM_small;
        MM_large = Sessions.MM_large;
        resp_small = Sessions.resp_small;
        resp_large = Sessions.resp_large;
        
        mixed_idx = [find(centered_idx); find(offcentered_idx)];
        
        % generate an upper triangular mask including the diagonal for the
        % centered pairs
        mask = triu(true(sum(centered_idx), sum(centered_idx)));
        
        % rsignal
        rsignal = corr(Sessions.MM_large','rows','all');
        rsignal_centered_versus_mixed = rsignal(centered_idx,:);
        rsignal_centered_versus_mixed = rsignal_centered_versus_mixed(:,mixed_idx);
        
        % pdist
        dists = pdist([Sessions.XYch(:,1),Sessions.XYch(:,2)],'euclidean');
        dists = squareform(dists);
        dist_centered_versus_mixed = dists(centered_idx,:);
        dist_centered_versus_mixed = dist_centered_versus_mixed(:,mixed_idx);
        
        % replace the upper triangle of centered pairs with nan
        rsignal_centered_versus_mixed(mask) = NaN;
        dist_centered_versus_mixed(mask) = NaN;
    end
    
    % rsc sorted by firing rate and distance
    count = 0;
    CC_small = NaN(NIm,N,N);
    CC_large = NaN(NIm,N,N);
    for i_image = 1:NIm
        count = count + 1;
        
        %  responsive cells per image
        if i_session ~= 9 % utah array thresholds
            thr1 = spontaneous;
            thr2 = max((spontaneous + 1 * stdspontaneous) * stim_blk_ratio ,0.1);
        else % neuropixel thresholds
            thr1 = max(spontaneous,0.1);
            thr2 = max((spontaneous + 0.1 * stdspontaneous),0.1);
        end
        
        resp_idx = zeros(N(1),1);
        resp_idx(centered_idx,1)  = (MM_small(centered_idx,i_image) >= thr2(centered_idx));
        resp_idx(offcentered_idx,1)  = (MM_small(offcentered_idx,i_image) <= thr1(offcentered_idx)) & (MM_large(offcentered_idx,i_image) >= thr2(offcentered_idx));
        
        tmp = squeeze(resp_small(:,i_image,:))';
        tmp(:,~resp_idx) = NaN;
        
        % make diagonal to zero
        tmp_corr = corr(tmp,'rows','all');
        tmp_corr(eye(size(tmp_corr)) == 1) = NaN;
        
        CC_small(count,:,:) = tmp_corr;
        
        tmp = squeeze(resp_large(:,i_image,:))';
        tmp(:,~resp_idx) = NaN;
        
        % make diagonal to zero
        tmp_corr = corr(tmp,'rows','all');
        tmp_corr(eye(size(tmp_corr)) == 1) = NaN;
        
        CC_large(count,:,:) = tmp_corr;
        
        % mixed
        tmpSmixed = squeeze(CC_small(count,centered_idx,mixed_idx));
        tmpSmixed(mask) = NaN;
        
        tmpLmixed = squeeze(CC_large(count,centered_idx,mixed_idx));
        tmpLmixed(mask) = NaN;
        
        CC_small_session = [CC_small_session; (tmpSmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
        CC_large_session = [CC_large_session; (tmpLmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
        
        SS_session = [SS_session; rsignal_centered_versus_mixed(isfinite(tmpSmixed(:) .* tmpLmixed(:)))];
        DD_session = [DD_session; dist_centered_versus_mixed(isfinite(tmpSmixed(:) .* tmpLmixed(:)))];
        
        CC_small_all = [CC_small_all; (tmpSmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
        CC_large_all = [CC_large_all; (tmpLmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
        
        SS_all = [SS_all; rsignal_centered_versus_mixed(isfinite(tmpSmixed(:) .* tmpLmixed(:)))];
        DD_all = [DD_all; dist_centered_versus_mixed(isfinite(tmpSmixed(:) .* tmpLmixed(:)))];
    end
end

% rsc sorted by rsignal and distance
bins_rsignal = 1 + eps:-2/nbins:-0.5 - eps;

dists_bins = 0 - eps:(100/nbins):100 + eps;
dists_bins(1) = 0;

CC_drsc_sorted_by_rsignal_dist = cell(nlocs,nrignals);
SS_sorted_by_rsignal_dist = cell(nlocs,nrignals);
DD_sorted_by_rsignal_dist = cell(nlocs,nrignals);

for i_bin_rsignal = 1 : nrignals
    for i_bin_dist = 1 : nlocs
        
        bin_idx = (dists_bins(i_bin_dist) <= DD_all) & (dists_bins(i_bin_dist+1) > DD_all) & ...
            (bins_rsignal(i_bin_rsignal) >= SS_all) & (bins_rsignal(i_bin_rsignal+1) < SS_all);
        
        CC_drsc_sorted_by_rsignal_dist{i_bin_dist,i_bin_rsignal} = CC_small_all(bin_idx) - CC_large_all(bin_idx);
        
        % rsignal
        SS_sorted_by_rsignal_dist{i_bin_dist, i_bin_rsignal} = SS_all(bin_idx);
        
        % distance
        DD_sorted_by_rsignal_dist{i_bin_dist, i_bin_rsignal} = DD_all(bin_idx);
    end
    
end

% drsc mean
neural_drsc = cellfun(@(M) (nanmean(M)),CC_drsc_sorted_by_rsignal_dist);

% rsignal mean
SS_mean = cellfun(@(M) (nanmean(M)),SS_sorted_by_rsignal_dist);

% dist mean
DD_mean = cellfun(@(M) (nanmean(M)),DD_sorted_by_rsignal_dist);

% check if the number of entry in CC matrix is lower than a threshold: 15
% entries
tmp_logical = cellfun(@(M) length(M)> min_entry_thr, CC_drsc_sorted_by_rsignal_dist);

%% linear regression
rs = SS_mean(tmp_logical);
d = DD_mean(tmp_logical);

drsc = neural_drsc(tmp_logical);

rs_normalized = (rs - mean(rs(:))) / std(rs(:));
d_normalized = (d - mean(d(:))) / std(d(:));
drsc_normalized = (drsc - mean(drsc(:))) / std(drsc(:));

tbl = table(rs_normalized, d_normalized, drsc_normalized);

lm = fitlm(tbl, 'linear');

disp(lm)

% Full model with both predictors
fullModel = fitlm(tbl, 'drsc_normalized ~ rs_normalized + d_normalized');

% Reduced model excluding rsignal
reducedModel_d = fitlm(tbl, 'drsc_normalized ~ d_normalized');

% Reduced model excluding distance
reducedModel_rs = fitlm(tbl, 'drsc_normalized ~ rs_normalized');

% Calculate change in R-squared
deltaR2_d = fullModel.Rsquared.Ordinary - reducedModel_d.Rsquared.Ordinary;
deltaR2_rs = fullModel.Rsquared.Ordinary - reducedModel_rs.Rsquared.Ordinary;

fprintf('Change in R-squared without rsignal: %f\n', deltaR2_d);
fprintf('Change in R-squared without dist: %f\n', deltaR2_rs);

%%
tmp = double(tmp_logical);
tmp(tmp == 0) = NaN;
neural_drsc = neural_drsc .* tmp;

% measure number of cases
selected_entries = CC_drsc_sorted_by_rsignal_dist(tmp_logical);
ncase = length(cell2mat(selected_entries));


% flip distance dimention
neural_drsc = flipud(neural_drsc);

save neural_drsc_20_locs_15_rsignal_bins neural_drsc ncase