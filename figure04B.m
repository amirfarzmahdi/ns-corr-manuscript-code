% description: noise correlation for small and large natural images,
% across sessions
% author: Amirhossein Farzmahdi
% last update: November 15th 2024

clear
close all
clc

% load aggregated data
load('aggregate_data.mat')

% add path
addpath(genpath('/functions/'))

% settings
nsession = 9; % 7: NN 2015 data, 1: anes136, 1: anes150
nbootstrap = 1000;
nbin_hist = 20;

colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980];
ffont = 8;
tfont = 6;

[xmin, ymin] = deal(0);
[xmax, ymax] = deal(0.25);

sbj_colors = [  100 , 92,  98;
                177, 166, 159;
                177, 166, 159;
                177, 166, 159;
                177, 166, 159;
                177, 166, 159;
                177, 166, 159;
                158, 124, 109;
                140, 100, 100] /255;
    
% adjust session 6 noise corr for breaking the x-axis
cc_cent_small{1,6} = cc_cent_small{1,6} - 0.15;
cc_cent_large{1,6} = cc_cent_large{1,6} - 0.15;
cc_mixed_large{1,6} = cc_mixed_large{1,6} - 0.15;
    
% centered
cc_cent_small_all = cell2mat(cc_cent_small(:));
cc_cent_large_all = cell2mat(cc_cent_large(:));

cc_mixed_small_all = cell2mat(cc_mixed_small(:));
cc_mixed_large_all = cell2mat(cc_mixed_large(:));

mm_cent_small_all = cell2mat(mm_cent_small(:));
mm_cent_large_all = cell2mat(mm_cent_large(:));

mm_mixed_small_all = cell2mat(mm_mixed_small(:));
mm_mixed_large_all = cell2mat(mm_mixed_large(:));

[selected_samples_data1,selected_samples_data2,new_edges_centered,new_counts_centered] = mean_matched_samples(mm_cent_small_all,mm_cent_large_all,nbootstrap,nbin_hist);
CC_small_bsamples_centered = mean(cc_cent_small_all(selected_samples_data1),1);
CC_large_bsamples_centered = mean(cc_cent_large_all(selected_samples_data2),1);

% mixed
[selected_samples_data1,selected_samples_data2,new_edges_mixed,new_counts_mixed] = mean_matched_samples(mm_mixed_small_all,mm_mixed_large_all,nbootstrap,nbin_hist);
CC_small_bsamples_mixed = mean(cc_mixed_small_all(selected_samples_data1),1);
CC_large_bsamples_mixed = mean(cc_mixed_large_all(selected_samples_data2),1);

% calculate mean and standard error for each row
sem = @(x) std(x) / sqrt(length(x));

m_session_small_centered = cellfun(@mean, cc_cent_small);
m_session_large_centered = cellfun(@mean, cc_cent_large);
std_session_small_centered = cellfun(sem, cc_cent_small);
std_session_large_centered = cellfun(sem, cc_cent_large);

m_session_small_mixed = cellfun(@mean, cc_mixed_small);
m_session_large_mixed = cellfun(@mean, cc_mixed_large);
std_session_small_mixed = cellfun(sem, cc_mixed_small);
std_session_large_mixed = cellfun(sem, cc_mixed_large);

%% plot section
figure
set(gcf,'Units','inches','Position',[0 0 3,3.3],'color',[1 1 1]);
ha = tight_subplot(1,1,[0 0],[0.13 0.04],[0.15 0.04]);
axes(ha(1));

% create scatter plot
hold on

for i_sbj = 1: nsession
    % centered
    scatter(m_session_small_centered(i_sbj), m_session_large_centered(i_sbj), 100, 'filled','MarkerFaceColor',sbj_colors(i_sbj,:),'Marker','o','MarkerFaceAlpha',1,'MarkerEdgeColor','w');
    errorbar(m_session_small_centered(i_sbj), m_session_large_centered(i_sbj), std_session_large_centered(i_sbj), 'horizontal', 'LineStyle', 'none', 'Color', 'k', 'CapSize',0, 'LineWidth', 0.7);
    errorbar(m_session_small_centered(i_sbj), m_session_large_centered(i_sbj), std_session_small_centered(i_sbj), 'vertical', 'LineStyle', 'none', 'Color', 'k', 'CapSize',0, 'LineWidth', 0.7);
    
    % mixed
    scatter(m_session_small_mixed(i_sbj), m_session_large_mixed(i_sbj), 100, 'filled','MarkerFaceColor',sbj_colors(i_sbj,:),'Marker','^','MarkerFaceAlpha',1,'MarkerEdgeColor','w');
    errorbar(m_session_small_mixed(i_sbj), m_session_large_mixed(i_sbj), std_session_large_mixed(i_sbj), 'horizontal', 'LineStyle', 'none', 'Color', 'k', 'CapSize',0, 'LineWidth', 0.7);
    errorbar(m_session_small_mixed(i_sbj), m_session_large_mixed(i_sbj), std_session_small_mixed(i_sbj), 'vertical', 'LineStyle', 'none', 'Color', 'k', 'CapSize',0, 'LineWidth', 0.7);
end

line([xmin,xmax],[ymin,ymax],'Color',[0.7,0.7,0.7],'LineStyle','--','LineWidth',1)

% gap anotation
set(0,'DefaultTextInterpreter','none')

text(0.17,0,'//','fontsize',6,'FontWeight','bold','FontName','Arial')
text(-0.0025,0.13,('\\'),'fontsize',6,'FontWeight','bold','FontName','Arial')

set(0,'DefaultTextInterpreter','tex')

% legend('centered',...
%        'mixed',...
%        'Location','north') 

% legend boxoff

xlim([xmin,xmax]);
ylim([ymin,ymax]);
xticks([0 0.05 0.1 0.15 0.2, 0.25]) 
yticks([0 0.05 0.1 0.15 0.2, 0.25]) 
xticklabels([0 0.05 0.1 0.15 0.33, 0.38]) 
yticklabels([0 0.05 0.1 0.30 0.35, 0.4])
ha(1).TickDir = 'out';
ha(1).TickLength = [0.005,0.005];
ha(1).FontName = 'Arial';
ha(1).FontSize = ffont;
ha(1).Clipping  = 'off';
ha(1).LineWidth = 1;

xlabel('r_{sc} small')
ylabel('r_{sc} large')

box off

% centered inset
gca;
hi = axes('Position',[.8 .22 .1 .125]);

% small
s = swarmchart(0.2*ones(length(CC_small_bsamples_centered),1), CC_small_bsamples_centered,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(1,:),'MarkerFaceColor',colors(1,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

hold on

% large
s = swarmchart(0.4*ones(length(CC_large_bsamples_centered),1), CC_large_bsamples_centered,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(2,:),'MarkerFaceColor',colors(2,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

[~, p_values] = ttest2(CC_small_bsamples_centered,CC_large_bsamples_centered);
groups={[0.2,0.4]};

H = sigstar(groups,p_values);
set(H,'LineWidth',0.5)

xlim([0.1,0.4])
y_inset = [0.08, 0.10, 0.12];

ylim([y_inset(1),y_inset(end)])
xticks([0.2,0.4])
xticklabels([{'small'};{'large'}])
yticks(y_inset)
text(-0.01,y_inset(end)+0.012,'mean-matched','FontName','Arial','FontSize',6)

yticklabels(y_inset)
hi.TickDir = 'out';
hi.TickLength = [0.02,0.02];
hi.FontSize = tfont;
hi.FontName = 'Arial';
hi.Clipping = 'off';

% mixed inset
gca;
hi = axes('Position',[.22 .8 .1 .125]);

% small
s = swarmchart(0.2*ones(length(CC_small_bsamples_mixed),1), CC_small_bsamples_mixed,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(1,:),'MarkerFaceColor',colors(1,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

hold on

% large
s = swarmchart(0.4*ones(length(CC_large_bsamples_mixed),1), CC_large_bsamples_mixed,'Marker','.','MarkerFaceAlpha',0.6,'MarkerEdgeAlpha',0.4,'MarkerEdgeColor',colors(2,:),'MarkerFaceColor',colors(2,:));
s.XJitterWidth = 0.1;
s.YJitterWidth = 0.2;
s.SizeData = 3;

[~, p_values] = ttest2(CC_small_bsamples_mixed,CC_large_bsamples_mixed);
groups={[0.2,0.4]};

H = sigstar(groups,p_values);
set(H,'LineWidth',0.5)

xlim([0.1,0.4])
y_inset = [0.04,0.06,0.08];


ylim([y_inset(1),y_inset(end)])
xticks([0.2,0.4])
xticklabels([{'small'};{'large'}])
yticks(y_inset)
text(-0.01,y_inset(end)+0.012,'mean-matched','FontName','Arial','FontSize',6)

yticklabels(y_inset)
hi.TickDir = 'out';
hi.TickLength = [0.02,0.02];
hi.FontSize = tfont;
hi.FontName = 'Arial';
hi.Clipping = 'off';

savetopdf(gcf,3,3.3,'figure04B_R1')
