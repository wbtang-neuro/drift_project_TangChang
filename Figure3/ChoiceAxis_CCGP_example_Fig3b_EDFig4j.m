clc
clear all
close all
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\libsvm-3.22')) % SVM toolbox
defaultGraphicsSetttings % graphics settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP8'; % animal prefix
day = 17; %day
epochs = [2,4,6]; % behavioral epochs
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% set parameters
tr_cond = 2; % number of trajectory types, 2, left + right
pos_interp = (0:0.005:1)';
choice_point = 0.37; % choice point is at the 37% of the whole trajectory
validbin_ids = find(pos_interp <= choice_point); % for choice analysis, we only consider the center stem
%% load data
% load files
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info

% load CA1 PYR spikes
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");

% load the trial-by-trial ratemaps
load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load firing rate maps
%% gather rate maps
trial_conditions = behavior.trialConds; % trajectory types/conditions

spike_matrix_animal = [];
spike_matrix_label = [];
alltr_count = 0;
dCA1_cellnum = length(spikes.UID);
cellnum = length(firingMaps_trials);
ratemap_var_all = [];
for i = 1:cellnum
    ratemap_var_cell = [];
    trial_count = [];
    epoch_count = [];
    condition_count = [];
    condition_label = [];

    for epoch = epochs
        epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time
        trialid_ep = find(behavior.trials(:,1) >= epochtimes(1) &  behavior.trials(:,2) <= epochtimes(2));
        trial_correct  = behavior.trialcorrect(trialid_ep);
        trialtime = mean(behavior.trials(trialid_ep,:),2);
        trial_condition = behavior.trialConds(trialid_ep);

        alltr_cond = find(trial_correct > -1);% take all trials
     

        % rescale just for visualization purposes (to match the colormap
        % lengths)
        tr_no = trialid_ep(alltr_cond);
        trial_count = [trial_count;tr_no];
        condition_tmp = trial_condition(alltr_cond);
        cond1 = find(condition_tmp == 1);
    
        condition_rescale = zeros(size(condition_tmp));
        condition_rescale(cond1) = (tr_no(cond1)-min(tr_no(cond1)))./(max(tr_no(cond1)) - min(tr_no(cond1)))*0.5;
        cond2 = find(condition_tmp == 2);
       
        condition_rescale(cond2) = (tr_no(cond2)-min(tr_no(cond2)))./(max(tr_no(cond2)) - min(tr_no(cond2)))*0.5 +0.5;
        condition_count = [condition_count;condition_rescale+ epoch/2-1];
        condition_label = [condition_label;condition_tmp];

        epoch_count = [epoch_count;epoch*ones(size(tr_no))];
     
        for tr_count = 1:length(tr_no)
            tr = tr_no(tr_count);
            temp1 = firingMaps_trials{i}.rateMaps(tr,:);
            temp1(isnan(temp1)) = 0;
            linfield_hp = interp1(0:1/99:1,temp1,pos_interp,'nearest');
            linfield_hp = (linfield_hp(validbin_ids)); %before CP
            ratemap_var_cell = [ratemap_var_cell,linfield_hp];
        end
    end
    ratemap_var_all = [ratemap_var_all;ratemap_var_cell];

end
PVcorr = corr(ratemap_var_all);
%% MDS
dm = 1 - PVcorr; % dissimilarity
[mdssim stressValue] = mdscale(dm,3,'criterion','metricstress','start','cmdscale');
%% plot MDS results
epoch1_trno = find(epoch_count == epochs(1));
epoch2_trno = find(epoch_count == epochs(2));
epoch3_trno = find(epoch_count == epochs(3));
% define colormap
c = [103	0       31;
    178     24      43;
    214     96      77;
    244     165     130;
    253     219     199;
    209     229     240;
    146     197     222;
    67	    147     195;
    33      102     172;
    5       48      97];
c = flipud(c/255);
traj_colormap = [c(2,:);c(end-1,:);c(3,:);c(end-2,:);c(4,:);c(end-3,:)];
% define dot sizes for different trials
size_vec = [(3:-(3-0.25)/(length(epoch1_trno)-1):0.25)'; (3:-(3-0.25)/(length(epoch2_trno)-1):0.25)';(3:-(3-0.25)/(length(epoch3_trno)-1):0.25)'];

figure
hold on
mdssim1 = mdssim(epoch1_trno,:);
mdssim2 = mdssim(epoch2_trno,:);
mdssim3 = mdssim(epoch3_trno,:);
scatter(mdssim(:,1),mdssim(:,3),60*size_vec,condition_count,'filled')
colormap(traj_colormap)
grid on
xlim([-0.8,0.8])
axis square 
colorbar
%% Demonstrate CCGP using a linear SVM decoder. build the model
trnum1 = length(epoch3_trno);
trnum2 = length(epoch1_trno);

% train on the last epoch data and predict the first epoch data
training_data = mdssim3;
training_label = condition_label(epoch3_trno);
dat_all.Features = training_data';
dat_all.trajlabel = training_label';
test_data =  mdssim1;
test_label = condition_label(epoch1_trno);

dat_all.Features_test = test_data';
dat_all.trajlabel_test = test_label';

figure,
scatter3(training_data(:,1),training_data(:,2),training_data(:,3),50*ones(size(training_label)),training_label-0.1,'LineWidth',2)
hold on
scatter3(test_data(:,1),test_data(:,2),test_data(:,3),60*ones(size(test_label)),test_label+2.1,'filled')
traj_colormap = [c(2,:);c(end-1,:);c(3,:);c(end-2,:);c(4,:);c(end-3,:)];

colormap(traj_colormap)
caxis([1,4])
%% predictions using linear SVMs
[accuracy_training,accuracy_test,accuracy_shuffle,p_value,~] = decoding_CCGP_linearSVM_simple(dat_all,1);
accuracy_shuf_95th =  prctile(accuracy_shuffle,95);