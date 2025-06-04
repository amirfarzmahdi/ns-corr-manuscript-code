% description: noise correlation verus rsignal and distance
% for small and large natural images, nn2015 data,
% centered and mixed (one example session: 7)
% author: Amirhossein Farzmahdi
% last update: June 20th 2024

clear
close all
clc

% load example session
data_dir = '/data/NN2015/';
i_session = 7;
load([data_dir '0' num2str(i_session) '.mat'])

% add path
addpath(genpath('/functions/'))

% settings
reps = 20; % number of repeats
colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980];
latency = 50;
nbin_rsignal = 5;
stimwindow = [1 105];
blkwindow = [161 210];
stim_blk_ratio =  (diff(stimwindow) +1)/ (diff(blkwindow) + 1);
npanel = 4;
ffont = 8;
lgfont = 8;
tfont = 6;
vis_size = 50;

% shift train window for response latency
resp_train = resp_train(:,:,:,1:end-1);
resp_train_blk = resp_train_blk(:,:,:,1:end-1);
tmp = resp_train; % [#neurons #stimuli #repeats #millisecond]
tmp(:,:,:,1:end-latency) = resp_train(:,:,:,latency+1:end);
tmp(:,:,:,end-latency+1:end) = resp_train_blk(:,:,:,1:latency);
resp_train = tmp;

% small and large spike count
resp = sum(resp_train,4,'omitnan'); % [#neurons #stimuli #repeats]
tmp = squeeze(resp(:,1:(2*9*30),:));

tmp = reshape(tmp,size(tmp,1),2,9,30,reps);
resp_small =  squeeze(tmp(:,1,:,:,:));
resp_large =  squeeze(tmp(:,2,:,:,:));

resp_small = reshape(resp_small,[size(resp_small,1),size(resp_small,2)*size(resp_small,3),size(resp_small,4)]);
resp_large = reshape(resp_large,[size(resp_large,1),size(resp_large,2)*size(resp_large,3),size(resp_large,4)]);

NIm = size(resp_large,2);
N = size(resp_large,1);
T = size(resp_large,3);

MM_small = mean(resp_small,3,'omitnan'); % small natural images
MM_large = mean(resp_large,3,'omitnan'); % large natural images

% spontaneous response
tmp = nansum(resp_train_blk(:,1:2*9*30,:,blkwindow(1):blkwindow(2)),4);
spontaneous = mean(tmp(:,:),2,'omitnan');
stdspontaneous = std(tmp(:,:),[],2,'omitnan');

% centered index
centered_idx = ~isnan(RF_SPATIAL(:,1)) &...
    sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 25;

offcentered_idx = ~isnan(RF_SPATIAL(:,1)) &...
    sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) >= 30 &...
    sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 150;

centered_tril_mat = tril(ones(sum(centered_idx)),-1);
centered_tril_mat(centered_tril_mat==0) = NaN;

offcentered_tril_mat = tril(ones(sum(offcentered_idx)),-1);
offcentered_tril_mat(offcentered_tril_mat==0) = NaN;

% rsignal
rsignal = corr(MM_large','rows','all');

rsignal_centered = rsignal(centered_idx,centered_idx);
rsignal_centered = rsignal_centered .* centered_tril_mat;

rsignal_offcentered = rsignal(offcentered_idx,offcentered_idx);
rsignal_offcentered = rsignal_offcentered .* offcentered_tril_mat;

rsignal_mixed = rsignal(centered_idx,offcentered_idx);

% pdist
dists = pdist([RF_SPATIAL(:,1),RF_SPATIAL(:,2)],'euclidean');
dists = squareform(dists);

dist_centered = dists(centered_idx,centered_idx);
dist_centered = dist_centered .* centered_tril_mat;

dist_mixed = dists(centered_idx,offcentered_idx);

% rsc sorted by firing rate and distance
CC_small = NaN(NIm,N,N);
CC_large = NaN(NIm,N,N);
MM_FF_small = NaN(NIm,N,N);
MM_FF_large = NaN(NIm,N,N);

CC_small_centered = [];
CC_large_centered = [];
CC_small_mixed = [];
CC_large_mixed = [];

SS_centered = [];
SS_mixed = [];
DD_mixed = [];
DD_centered = [];

MM_small_centered = [];
MM_large_centered = [];
MM_small_mixed = [];
MM_large_mixed = [];

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
    CC_small_centered = [CC_small_centered; (tmpScent(isfinite(tmpScent(:) .* tmpLcent(:))))];
    CC_large_centered = [CC_large_centered; (tmpLcent(isfinite(tmpScent(:) .* tmpLcent(:))))];
    
    SS_centered = [SS_centered;rsignal_centered(isfinite(tmpScent(:) .* tmpLcent(:)))];
    DD_centered = [DD_centered; dist_centered(isfinite(tmpScent(:) .* tmpLcent(:)))];
    
    % mixed
    tmpSmixed = squeeze(CC_small(count,centered_idx,offcentered_idx));
    tmpLmixed = squeeze(CC_large(count,centered_idx,offcentered_idx));
    CC_small_mixed = [CC_small_mixed; (tmpSmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
    CC_large_mixed = [CC_large_mixed; (tmpLmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
    
    SS_mixed = [SS_mixed; rsignal_mixed(isfinite(tmpSmixed(:) .* tmpLmixed(:)))];
    DD_mixed = [DD_mixed; dist_mixed(isfinite(tmpSmixed(:) .* tmpLmixed(:)))];
    
    sNeurons_cent = find(centered_idx);
    sNeurons_offcent = find(offcentered_idx);
    
    % mixed
    [cent_idx,offcent_idx] = find(isfinite(tmpSmixed .* tmpLmixed));
    
    mean_pair_small = mean([MM_small(sNeurons_offcent(offcent_idx),count),...
        MM_small(sNeurons_cent(cent_idx),count)],2);
    mean_pair_large = mean([MM_large(sNeurons_offcent(offcent_idx),count),...
        MM_large(sNeurons_cent(cent_idx),count)],2);
    
    MM_small_mixed = [MM_small_mixed; mean_pair_small];
    MM_large_mixed = [MM_large_mixed; mean_pair_large];
    
    % centered
    [cent_idx1, cent_idx2] = find(isfinite(tmpScent .* tmpLcent));
    mean_pair_small = mean([MM_small(sNeurons_cent(cent_idx1),count),...
        MM_small(sNeurons_cent(cent_idx2),count)],2);
    mean_pair_large = mean([MM_large(sNeurons_cent(cent_idx1),count),...
        MM_large(sNeurons_cent(cent_idx2),count)],2);
    
    MM_small_centered = [MM_small_centered; mean_pair_small];
    MM_large_centered = [MM_large_centered; mean_pair_large];
end

% rsc sorted by rsignal
bins_centered = linspace(1+eps,-0.2,nbin_rsignal);
bins_mixed = linspace(1+eps,-0.2,nbin_rsignal);

CC_small_sorted_by_rsignal_centered = cell(nbin_rsignal-1,1);
CC_large_sorted_by_rsignal_centered = cell(nbin_rsignal-1,1);
CC_small_sorted_by_rsignal_mixed = cell(nbin_rsignal-1,1);
CC_large_sorted_by_rsignal_mixed = cell(nbin_rsignal-1,1);

SS_centered_bins =  cell(nbin_rsignal-1,1);
SS_mixed_bins =  cell(nbin_rsignal-1,1);

DD_centered_bins =  cell(nbin_rsignal-1,1);
DD_mixed_bins =  cell(nbin_rsignal-1,1);

for i_bin = 1 : nbin_rsignal - 1
    bin_idx_centered = find((bins_centered(i_bin) > SS_centered) & (bins_centered(i_bin+1) < SS_centered));
    if length(bin_idx_centered) > 1
        CC_small_sorted_by_rsignal_centered{i_bin} = [CC_small_sorted_by_rsignal_centered{i_bin};CC_small_centered(bin_idx_centered)];
        CC_large_sorted_by_rsignal_centered{i_bin} = [CC_large_sorted_by_rsignal_centered{i_bin};CC_large_centered(bin_idx_centered)];
        SS_centered_bins{i_bin} = [SS_centered_bins{i_bin};SS_centered(bin_idx_centered)];
    end
    
    bin_idx_mixed = find((bins_mixed(i_bin) > SS_mixed) & (bins_mixed(i_bin+1) <= SS_mixed));
    if length(bin_idx_mixed) > 1
        CC_small_sorted_by_rsignal_mixed{i_bin} = [CC_small_sorted_by_rsignal_mixed{i_bin};CC_small_mixed(bin_idx_mixed)];
        CC_large_sorted_by_rsignal_mixed{i_bin} = [CC_large_sorted_by_rsignal_mixed{i_bin};CC_large_mixed(bin_idx_mixed)];
        SS_mixed_bins{i_bin} = [SS_mixed_bins{i_bin};SS_mixed(bin_idx_mixed)];
    end
    
end

DD_bins = [DD_centered;DD_mixed];
% rsc sorted by distance
dists_bins_centered = min(DD_bins)-eps:19.25:max(DD_bins)+eps;
dists_bins_mixed = min(DD_bins)-eps:19.25:max(DD_bins)+eps;
dists_bins_centered(end) = max(DD_bins)+eps;
dists_bins_mixed(end) = max(DD_bins)+eps;

CC_small_sorted_by_dist_centered = cell(nbin_rsignal-1,1);
CC_large_sorted_by_dist_centered = cell(nbin_rsignal-1,1);
CC_small_sorted_by_dist_mixed = cell(nbin_rsignal-1,1);
CC_large_sorted_by_dist_mixed = cell(nbin_rsignal-1,1);

for i_bin = 1 : nbin_rsignal - 1
    
    bin_idx_centered = find((dists_bins_centered(i_bin) < DD_centered) & (dists_bins_centered(i_bin+1) > DD_centered));
    if numel(bin_idx_centered) > 1
        CC_small_sorted_by_dist_centered{i_bin} = [CC_small_sorted_by_dist_centered{i_bin};CC_small_centered(bin_idx_centered)];
        CC_large_sorted_by_dist_centered{i_bin} = [CC_large_sorted_by_dist_centered{i_bin};CC_large_centered(bin_idx_centered)];
        DD_centered_bins{i_bin} = [DD_centered_bins{i_bin};DD_centered(bin_idx_centered)];
    end
    
    bin_idx_mixed = find((dists_bins_mixed(i_bin) < DD_mixed) & (dists_bins_mixed(i_bin+1) > DD_mixed));
    if numel(bin_idx_mixed) > 1
        CC_small_sorted_by_dist_mixed{i_bin} = [CC_small_sorted_by_dist_mixed{i_bin};CC_small_mixed(bin_idx_mixed)];
        CC_large_sorted_by_dist_mixed{i_bin} = [CC_large_sorted_by_dist_mixed{i_bin};CC_large_mixed(bin_idx_mixed)];
        
        DD_mixed_bins{i_bin} = [DD_mixed_bins{i_bin};DD_mixed(bin_idx_mixed)];
    end
    
end

%% plot
figure;
set(gcf,'Units','inches','Position',[0 0 3.5,3.5],'color',[1 1 1]);
hs = tight_subplot(2,2,[0.12 0.12],[0.11 0.02],[0.12 0.04]);

axes(hs(1));
CC_MM_small_centered_rsignal = cellfun(@(M) (nanmean(M)),CC_small_sorted_by_rsignal_centered);
CC_MM_large_centered_rsignal = cellfun(@(M) (nanmean(M)),CC_large_sorted_by_rsignal_centered);
CC_VV_small_centered_rsignal = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_small_sorted_by_rsignal_centered);
CC_VV_large_centered_rsignal = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_large_sorted_by_rsignal_centered);

SS_MM_centered = cellfun(@(M) (nanmean(M)), SS_centered_bins);
SS_VV_centered = cellfun(@(M) (nanstd(M)/sqrt(length(M))), SS_centered_bins);

x_edges = round(bins_centered,1);
x_cent_bins = x_edges(1:end-1) - ((x_edges(1) - x_edges(2))/2);
x_pos = 1:length(CC_MM_small_centered_rsignal);
x_pos_SS = x_pos + (SS_MM_centered' - x_cent_bins);

errorbar(x_pos_SS,CC_MM_small_centered_rsignal,CC_VV_small_centered_rsignal,CC_VV_small_centered_rsignal, SS_VV_centered, SS_VV_centered, 'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(1,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)
hold on
errorbar(x_pos_SS,CC_MM_large_centered_rsignal,CC_VV_large_centered_rsignal, CC_VV_large_centered_rsignal, SS_VV_centered, SS_VV_centered, 'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(2,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)

y_max = round(10 * max([CC_MM_small_centered_rsignal;CC_MM_large_centered_rsignal]))/10 + 0.1;
y_min = 0;
xticks(0.5:1:nbin_rsignal)
xticklabels(1-x_edges)
xtickangle(0)
yticks(y_min:0.1:y_max)
yticklabels(y_min:0.1:y_max)
xlim([0.5,nbin_rsignal-0.5])
ylim([y_min,y_max])

xlabel('tuning dissimilarity(1-r_{signal})')
ylabel('r_{sc}')
box off

% legend
lg = legend('small image','large image');
lg.FontSize = lgfont;
lg.FontName = 'Arial';
lg.Box = 'off';
lg.LineWidth = 0.5;

text(1.5,0.015,['ncase = ' addComma(sum(~isnan(SS_centered),'all'))],'FontName','Arial','FontSize',tfont)

axes(hs(2));

CC_MM_small_centered_dist = cellfun(@(M) (nanmean(M)),CC_small_sorted_by_dist_centered);
CC_MM_large_centered_dist = cellfun(@(M) (nanmean(M)),CC_large_sorted_by_dist_centered);
CC_VV_small_centered_dist = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_small_sorted_by_dist_centered);
CC_VV_large_centered_dist = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_large_sorted_by_dist_centered);

dists_bins_centered(1) = 0;
x_edges = round(dists_bins_centered/vis_size,1);

DD_MM_centered = cellfun(@(M) (nanmean(M/vis_size)),DD_centered_bins);
DD_VV_centered = cellfun(@(M) (nanstd(M/vis_size)/sqrt(length(M))),DD_centered_bins);

x_cent_bins = x_edges(1:end-1) - ((x_edges(1) - x_edges(2))/2);
x_pos = 1:length(CC_MM_small_centered_dist);
x_pos_DD = x_pos + (DD_MM_centered' - x_cent_bins);

errorbar(x_pos_DD,CC_MM_small_centered_dist,CC_VV_small_centered_dist, CC_VV_small_centered_dist, DD_VV_centered, DD_VV_centered, 'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(1,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)
hold on
errorbar(x_pos_DD,CC_MM_large_centered_dist,CC_VV_large_centered_dist, CC_VV_large_centered_dist, DD_VV_centered, DD_VV_centered, 'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(2,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)

xlim([0.5,nbin_rsignal-0.5])
ylim([y_min,y_max])
xticks(0.5:1:nbin_rsignal)
xticklabels(x_edges)
xtickangle(0)
yticks(y_min:0.1:y_max)
yticklabels(y_min:0.1:y_max)
xlabel('RF distance (deg)')
ylabel('')
box off

axes(hs(3));

CC_MM_small_mixed_rsignal = cellfun(@(M) (nanmean(M)),CC_small_sorted_by_rsignal_mixed);
CC_MM_large_mixed_rsignal = cellfun(@(M) (nanmean(M)),CC_large_sorted_by_rsignal_mixed);
CC_VV_small_mixed_rsignal = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_small_sorted_by_rsignal_mixed);
CC_VV_large_mixed_rsignal = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_large_sorted_by_rsignal_mixed);

SS_MM_mixed = cellfun(@(M) (nanmean(M)),SS_mixed_bins);
SS_VV_mixed = cellfun(@(M) (nanstd(M)/sqrt(length(M))),SS_mixed_bins);

x_edges = round(bins_mixed,1);
x_cent_bins = x_edges(1:end-1) - ((x_edges(1) - x_edges(2))/2);
x_pos = 1:length(CC_MM_small_mixed_rsignal);
x_pos_SS = x_pos + (SS_MM_mixed' - x_cent_bins);

errorbar(x_pos_SS,CC_MM_small_mixed_rsignal,CC_VV_small_mixed_rsignal,CC_VV_small_mixed_rsignal,SS_VV_mixed,SS_VV_mixed,'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(1,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)
hold on
errorbar(x_pos_SS,CC_MM_large_mixed_rsignal,CC_VV_large_mixed_rsignal,CC_VV_large_mixed_rsignal,SS_VV_mixed,SS_VV_mixed,'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(2,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)

y_min = 0;
xlim([0.5,nbin_rsignal-0.5])
ylim([y_min,y_max])
xticks(0.5:1:nbin_rsignal)
xticklabels(1-x_edges)
xtickangle(0)
yticks(y_min:0.1:y_max)
yticklabels(y_min:0.1:y_max)
xlabel('tuning dissimilarity(1-r_{signal})')
ylabel('r_{sc}')
box off

text(1.5,0.015,['ncase = ' addComma(sum(~isnan(SS_mixed),'all'))],'FontName','Arial','FontSize',tfont)

axes(hs(4));
CC_MM_small_mixed_dist = cellfun(@(M) (nanmean(M)),CC_small_sorted_by_dist_mixed);
CC_MM_large_mixed_dist = cellfun(@(M) (nanmean(M)),CC_large_sorted_by_dist_mixed);
CC_VV_small_mixed_dist = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_small_sorted_by_dist_mixed);
CC_VV_large_mixed_dist = cellfun(@(M) (nanstd(M)/sqrt(length(M))),CC_large_sorted_by_dist_mixed);

DD_MM_mixed = cellfun(@(M) (nanmean(M/vis_size)), DD_mixed_bins);
DD_VV_mixed = cellfun(@(M) (nanstd(M/vis_size)/sqrt(length(M))), DD_mixed_bins);

dists_bins_mixed(1) = 0;
x_edges = round(dists_bins_mixed/vis_size,1);
x_cent_bins = x_edges(1:end-1) - ((x_edges(1) - x_edges(2))/2);
x_pos_DD = x_pos + (DD_MM_mixed' - x_cent_bins);

errorbar(x_pos_DD,CC_MM_small_mixed_dist,CC_VV_small_mixed_dist,CC_VV_small_mixed_dist,DD_VV_mixed,DD_VV_mixed, 'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(1,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)
hold on
errorbar(x_pos_DD,CC_MM_large_mixed_dist,CC_VV_large_mixed_dist,CC_VV_large_mixed_dist,DD_VV_mixed,DD_VV_mixed, 'Marker','o','MarkerFaceColor','none',...
    'MarkerEdgeColor',colors(2,:),'LineStyle','-','CapSize',0,'MarkerSize',4,'LineWidth',1)

xlim([0.5,nbin_rsignal-0.5])
ylim([y_min,y_max])
xticks(0.5:1:nbin_rsignal)
xticklabels(x_edges)
xtickangle(0)
yticks(y_min:0.1:y_max)
yticklabels(y_min:0.1:y_max)
xlabel('RF distance (deg)')
ylabel('')
box off

% general axes settings
for i_panel = 1:npanel
    hs(i_panel).TickDir = 'out';
    hs(i_panel).TickLength = [0.005,0.005];
    hs(i_panel).FontSize = ffont;
    hs(i_panel).FontName = 'Arial';
    hs(i_panel).XTickLabelRotation = 0;
    hs(i_panel).LineWidth = 1;
end

savetopdf(gcf,3.5,3.5,'figure05')