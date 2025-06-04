% description: noise correlation for small and large natural images,
% nn2015 data, centered and mixed (one example session: 7)
% author: Amirhossein Farzmahdi
% last update: November 15th 2024

clear
close all
clc

% load data
data_dir = '/data/NN2015/';
i_session = 7;
load([data_dir '0' num2str(i_session) '.mat'])

% add path
addpath(genpath('/functions/'));

% settings
reps = 20; % number of repeats
colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980];
nbin_hist = 20;
latency = 50;
nbootstrap = 1000;
stimwindow = [1 105];
blkwindow = [161 210];
stim_blk_ratio =  (diff(stimwindow) +1)/ (diff(blkwindow) + 1);
npanel = 4;
ffont = 8;
tfont = 6;

% plot settings
corner_pix_val = 255;
stim_center = [127.5, 127.5];

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

MM_small = mean(resp_small,3,'omitnan'); % small natural images
MM_large = mean(resp_large,3,'omitnan'); % large natural images

NIm = size(resp_large,2);
N = size(resp_large,1);
T = size(resp_large,3);

% spontaneous response
tmp = nansum(resp_train_blk(:,1:2*9*30,:,blkwindow(1):blkwindow(2)),4);
spontaneous = mean(tmp(:,:),2,'omitnan');
stdspontaneous = std(tmp(:,:),[],2,'omitnan');

% centered index
centered_idx = sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 25;

offcentered_idx = ~isnan(RF_SPATIAL(:,1)) &...
    sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) >= 30 &...
    sqrt(sum([RF_SPATIAL(:,1).^2,RF_SPATIAL(:,2).^2],2)) <= 150;

centered_tril_mat = tril(ones(sum(centered_idx)),-1);
centered_tril_mat(centered_tril_mat==0) = NaN;

offcentered_tril_mat = tril(ones(sum(offcentered_idx)),-1);
offcentered_tril_mat(offcentered_tril_mat==0) = NaN;

CC_small = NaN(NIm,N,N);
CC_large = NaN(NIm,N,N);
CC_small_centered = [];
CC_large_centered = [];
CC_small_mixed = [];
CC_large_mixed = [];

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
    
    % mixed
    tmpSmixed = squeeze(CC_small(count,centered_idx,offcentered_idx));
    tmpLmixed = squeeze(CC_large(count,centered_idx,offcentered_idx));
    CC_small_mixed = [CC_small_mixed; (tmpSmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
    CC_large_mixed = [CC_large_mixed; (tmpLmixed(isfinite(tmpSmixed(:) .* tmpLmixed(:))))];
    
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

% mean-match analysis
% centered
[selected_samples_data1,selected_samples_data2,new_edges_centered,new_counts_centered] = mean_matched_samples(MM_small_centered,MM_large_centered,nbootstrap,nbin_hist);
CC_small_bsamples_centered = mean(CC_small_centered(selected_samples_data1),1);
CC_large_bsamples_centered = mean(CC_large_centered(selected_samples_data2),1);

% mixed
[selected_samples_data1,selected_samples_data2,new_edges_mixed,new_counts_mixed] = mean_matched_samples(MM_small_mixed,MM_large_mixed,nbootstrap,nbin_hist);
CC_small_bsamples_mixed = mean(CC_small_mixed(selected_samples_data1),1);
CC_large_bsamples_mixed = mean(CC_large_mixed(selected_samples_data2),1);

%% plot
figure;
set(gcf,'Units','inches','Position',[0 0 3.5,3.5],'color',[1 1 1]);
hs = tight_subplot(2,2,[0.15 0.15],[0.1 0.02],[0.01 0.1]);

% plot cell location
axes(hs(1))
hold on
stim = double(imcrop(rgb2gray(imread('/Users/amirfarzmahdi/Downloads/figure_code/figure04/284.jpg')),[122,42,255,255])); % select a natural image

% window
L1 = 128; % large patch radius
L2 = 25;  % small patch radius
[X, Y]=meshgrid(1:2*L1);
Zlarge=double(sqrt((X-L1).^2 + (Y-L1).^2) < L1);
Zsmall=double(sqrt((X-L1).^2 + (Y-L1).^2) < L2);
window = cell(2,1);
for i=1:2
    window{1} = Zsmall;
    window{2} = Zlarge;
end
stim = stim .* window{2} + corner_pix_val * (1-window{2});

hb = imshow(stim/255);
hb.AlphaData = 0.8;

RFs = cat(2,[RF_SPATIAL(centered_idx,:) ; RF_SPATIAL(offcentered_idx,:)]);

scatter(RFs(:,1) + stim_center(1), RFs(:,2)  + stim_center(2), ...
    20,'filled','MarkerFaceColor',[0.9961,0.8510,0.4627],'MarkerEdgeColor',[0.5,0.5,0.5],'MarkerFaceAlpha',0.7,'Marker',"o",'LineWidth',0.15)


scatter(RFs(1:2,1) + stim_center(1), RFs(1:2,2)  + stim_center(2), ...
    20,'filled','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerFaceAlpha',1,'Marker',"o",'LineWidth',0.25)

viscircles([127.5,127.5],L2,'Color',colors(1,:),'LineStyle','-','LineWidth',1);
viscircles([127.5,127.5],127.5,'Color',colors(2,:),'LineStyle','-','LineWidth',1);

text(190,245,['n_{cent} = ' num2str(sum(centered_idx))],'Color',[0,0,0]/255,'FontName','Arial','FontSize',tfont)

xlim([0,256])
ylim([0,256])
axis off
xticks('')
xticklabels('')
yticks('')
yticklabels('')
box off

axes(hs(2));
bin_width = 0.05; % 3.5 * std(CC_small_centered)/(length(CC_small_centered))^(1/3);

hist_small = histogram(CC_small_centered,'Normalization','probability','BinWidth',bin_width,'FaceColor',colors(1,:),'EdgeColor','w','FaceAlpha',0.6,'EdgeAlpha',0.1);
hist_small_mean = mean(CC_small_centered);
hist_small_std = std(CC_small_centered)/sqrt(length(CC_small_centered));
hold on

hist_large = histogram(CC_large_centered,'Normalization','probability','BinWidth',bin_width,'FaceColor',colors(2,:),'EdgeColor','w','FaceAlpha',0.6,'EdgeAlpha',0.1);
hist_large_mean = mean(CC_large_centered);
hist_large_std = std(CC_large_centered)/sqrt(length(CC_large_centered));

max_bin_count = max([hist_small.Values,hist_large.Values]);

triangle_height = 1.05 * max_bin_count;

errorbar(hist_small_mean,triangle_height,hist_small_std,'horizontal','v','MarkerSize',6,'MarkerFaceColor',colors(1,:),'MarkerEdgeColor','w','CapSize',0)
errorbar(hist_large_mean,triangle_height,hist_large_std,'horizontal','v','MarkerSize',6,'MarkerFaceColor',colors(2,:),'MarkerEdgeColor','w','CapSize',0)

xlim([min([CC_small_centered;CC_large_centered]),max([CC_small_centered;CC_large_centered])])
ylim([0, 0.1])
xticks(-1:0.5:1)
xticklabels(-1:0.5:1)
xlabel('r_{sc}')
ylabel('proportion of cases')
box off

[~, p_values] = ttest2(CC_small_centered,CC_large_centered);
groups={[hist_small_mean,hist_large_mean]};

H = sigstar(groups,p_values);
set(H,'LineWidth',0.5)

hs(2).LineWidth = 1;

% inset
gca;
hi = axes('Position',[.84 .8 .1 .125]);

s = swarmchart(0.2*ones(length(CC_small_bsamples_centered),1), CC_small_bsamples_centered,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(1,:),'MarkerFaceColor',colors(1,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

hold on
s = swarmchart(0.4*ones(length(CC_large_bsamples_centered),1), CC_large_bsamples_centered,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(2,:),'MarkerFaceColor',colors(2,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

[~, p_values] = ttest2(CC_small_bsamples_centered,CC_large_bsamples_centered);
groups={[0.2,0.4]};

H = sigstar(groups,p_values);
set(H,'LineWidth',0.5)
xlim([0.1,0.4])
y_inset = [0,0.1,0.2];
ylim([y_inset(1),y_inset(end)])
xticks([0.2,0.4])
xticklabels([{'small'};{'large'}])
yticks(y_inset)
text(-0.02,y_inset(end)+0.05,'mean-matched r_{sc}','FontName','Arial','FontSize',tfont)

yticklabels(y_inset)
hi.TickDir = 'out';
hi.TickLength = [0.02,0.02];
hi.FontSize = tfont;
hi.FontName = 'Arial';
hi.Clipping = 'off';

%% mixed
% plot cell location
axes(hs(3))
hold on

% window
L1 = 128; % large patch radius
L2 = 25;  % small patch radius
[X, Y]=meshgrid(1:2*L1);
Zlarge=double(sqrt((X-L1).^2 + (Y-L1).^2) < L1);
Zsmall=double(sqrt((X-L1).^2 + (Y-L1).^2) < L2);
window = cell(2,1);
for i=1:2
    window{1} = Zsmall;
    window{2} = Zlarge;
end
hb = imshow(stim/255);
hb.AlphaData = 0.8;

scatter(RFs(:,1) + stim_center(1), RFs(:,2)  + stim_center(2), ...
    20,'filled','MarkerFaceColor',[0.9961,0.8510,0.4627],'MarkerEdgeColor',[0.5,0.5,0.5],'MarkerFaceAlpha',0.7,'Marker',"o",'LineWidth',0.15)
scatter(RFs([1,51],1) + stim_center(1), RFs([1,51],2)  + stim_center(2), ...
    20,'filled','MarkerFaceColor','k','MarkerEdgeColor','w','MarkerFaceAlpha',1,'Marker',"o",'LineWidth',0.25)

viscircles([127.5,127.5],L2,'Color',colors(1,:),'LineStyle','-','LineWidth',1);
viscircles([127.5,127.5],127.5,'Color',colors(2,:),'LineStyle','-','LineWidth',1);

text(190,245,['n_{cent} = ' num2str(sum(centered_idx))],'Color','k','FontName','Arial','FontSize',tfont)
text(190,260,['n_{offcent} ' num2str(sum(offcentered_idx))],'Color','k','FontName','Arial','FontSize',tfont)

xlim([0,256])
ylim([0,256])
axis off
xticks('')
xticklabels('')
yticks('')
yticklabels('')
box off

axes(hs(4));
bin_width = 0.05; % 3.5 * std(CC_small_centered)/(length(CC_small_centered))^(1/3);

hist_small = histogram(CC_small_mixed,'Normalization','probability','BinWidth',bin_width,'FaceColor',colors(1,:),'EdgeColor','k','FaceAlpha',0.6,'EdgeAlpha',0.1);
hist_small_mean = mean(CC_small_mixed);
hist_small_std = std(CC_small_mixed)/sqrt(length(CC_small_mixed));
hold on

hist_large = histogram(CC_large_mixed,'Normalization','probability','BinWidth',bin_width,'FaceColor',colors(2,:),'EdgeColor','k','FaceAlpha',0.6,'EdgeAlpha',0.1);
hist_large_mean = mean(CC_large_mixed);
hist_large_std = std(CC_large_mixed)/sqrt(length(CC_large_mixed));

max_bin_count = max([hist_small.Values,hist_large.Values]);

triangle_height = 1.05 * max_bin_count;

errorbar(hist_small_mean,triangle_height,hist_small_std,'horizontal','v','MarkerSize',6,'MarkerFaceColor',colors(1,:),'MarkerEdgeColor','w','CapSize',0)
errorbar(hist_large_mean,triangle_height,hist_large_std,'horizontal','v','MarkerSize',6,'MarkerFaceColor',colors(2,:),'MarkerEdgeColor','w','CapSize',0)

xlim([min([CC_small_mixed;CC_large_mixed]),max([CC_small_mixed;CC_large_mixed])])
ylim([0, 0.1])
xticks(-1:0.5:1)
xticklabels(-1:0.5:1)
xlabel('r_{sc}')
ylabel('proportion of cases')
box off

[~, p_values] = ttest2(CC_small_mixed,CC_large_mixed);
groups={[hist_small_mean,hist_large_mean]};

H = sigstar(groups,p_values);
set(H,'LineWidth',0.5)

hs(4).LineWidth = 1;

% inset
gca;
hi = axes('Position',[.84 .3 .1 .125]);

s = swarmchart(0.2*ones(length(CC_small_bsamples_mixed),1), CC_small_bsamples_mixed,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(1,:),'MarkerFaceColor',colors(1,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

hold on
s = swarmchart(0.4*ones(length(CC_large_bsamples_mixed),1), CC_large_bsamples_mixed,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(2,:),'MarkerFaceColor',colors(2,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

[~, p_values] = ttest2(CC_small_bsamples_mixed,CC_large_bsamples_mixed);
groups={[0.2,0.4]};

H = sigstar(groups,p_values);
set(H,'LineWidth',0.5)
xlim([0.1,0.4])
y_inset = [0.0,0.1,0.2];
ylim([y_inset(1),y_inset(end)])
xticks([0.2,0.4])
xticklabels([{'small'};{'large'}])
yticks(y_inset)
yticklabels(y_inset)
text(-0.02,y_inset(end)+0.05,'mean-matched r_{sc}','FontName','Arial','FontSize',tfont)
hi.TickDir = 'out';
hi.TickLength = [0.02,0.02];
hi.FontSize = tfont;
hi.FontName = 'Arial';
hi.Clipping = 'off';

% general axes settings
for i_panel = 1:npanel
    hs(i_panel).TickDir = 'out';
    hs(i_panel).TickLength = [0.005,0.005];
    hs(i_panel).FontSize = ffont;
    hs(i_panel).FontName = 'Arial';
    hs(i_panel).XTickLabelRotation = 0;
end

savetopdf(gcf,3.5,3.5,'figure04A_R1')
