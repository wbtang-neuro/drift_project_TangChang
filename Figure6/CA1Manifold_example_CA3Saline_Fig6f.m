clear all
close all
clc
%% add codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\slanCM\slanCM\')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\umapAndEppFileExchange (4.1)')) %UMAP toolbox
defaultGraphicsSetttings % graphic settings
%% define data
dir = 'U:\data\Zutshi_Neuron2021\CA3_mEC\'; % data folder
% Fig.4i, top panels, Saline
animalprefix = 'IZ33';
SesssionCondition = 'Saline'; %Final/Saline
prefix = 'IZ33_580um_210304_sess3';
dir = [dir,animalprefix,'/',SesssionCondition,'\',prefix,'\'];
epochs = [1,4]; % behavioral epochs
%% set parameters
conditions = 2; % number of trajectory types, 2, left + right
pos_interp = (0:0.005:1)';
cellthresh = 5; % threshold for number of cell active in a time bin
%% load data
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1"); % CA1 PYR spikes
%% get the spike matrix
trial_conditions = behavior.trials.visitedArm + 1;  % label as 1 and 2
%-----load the trial-by-trial ratemaps-----%
load(fullfile(dir,  [prefix,'.firingMapsTrials.cellinfo.mat']));% load firing rate maps

spike_matrix_animal = []; % reset
spike_matrix_label = [];
spike_matrix_ep = [];

alltr_count = 0;
dCA1_cellnum = length(spikes.UID);
cellnum = length(firingMaps_trials);

for epoch = epochs
    epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time
    trialid_ep = find(behavior.trials.startPoint(:,1) >= epochtimes(1) &  behavior.trials.startPoint(:,2) <= epochtimes(2));
    trialtime = mean(behavior.trials.startPoint(trialid_ep,:),2);
    trial_condition = trial_conditions(trialid_ep);

    for tr_count = 1:length(trialid_ep)
        alltr_count = alltr_count +1;
        cell_count = 0;
        tr = trialid_ep(tr_count);
        tr_cond = trial_conditions(tr);

        if tr_cond == 2
            temp_label = pos_interp(end:-1:1) + tr_cond - 1;
        else
            temp_label = pos_interp + tr_cond - 1;
        end
        temp_ep = epoch * ones(size(temp_label));

        temp = zeros(length(pos_interp),dCA1_cellnum); % reset

        for i = 1:cellnum
            if ismember(firingMaps_trials{i}.UID, spikes.UID) % dorsal cells
                cell_count = cell_count + 1;
                temp1 = firingMaps_trials{i}.rateMaps(tr,:);
                if ~isempty(temp1)
                    temp1(isnan(temp1)) = 0;
                    linfield_hp = interp1(0:1/99:1,temp1,pos_interp,'nearest');
                else
                    linfield_hp = zeros(size(pos_interp));
                end
                temp(:,cell_count) = linfield_hp';
            end
        end

        spike_matrix_animal = [spike_matrix_animal;temp];
        spike_matrix_label = [spike_matrix_label;temp_label,...
            trial_condition(tr_count)*ones(size(temp_label)),...
            alltr_count*ones(size(temp_label)),trialtime(tr_count)*ones(size(temp_label))];
        spike_matrix_ep = [spike_matrix_ep;temp_ep];

    end
end
%% remove bins without enough spikes, apply cell threshold
validid = [];
for i = 1:length(spike_matrix_animal(:,1))
    temp = spike_matrix_animal(i,:);
    id = find(temp > 0);
    if length(id) >= cellthresh
        validid = [validid;i];
    end
end

spike_matrix_animal = spike_matrix_animal(validid,:);
spike_matrix_animal = zscore(spike_matrix_animal')';
spike_matrix_label = spike_matrix_label(validid,:);
spike_matrix_ep = spike_matrix_ep(validid);
%% run UMAP
spike_umap = run_umap(spike_matrix_animal,'min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3,'template_file','CA3_Saline_template.mat'); % use the previously acquired template file for repreducibility, you can also run it anew
%% exclude the 10% start and end of each trajectory to avoid edging errors
spike_matrix_label_pos = spike_matrix_label(:,1);
pos_lb = 0.1;
pos_hb = 0.9;
validlabels_id = find( (spike_matrix_label_pos > pos_lb & spike_matrix_label_pos < pos_hb) | (spike_matrix_label_pos > pos_lb +1 & spike_matrix_label_pos < pos_hb +1));
auxiliary_variables  = spike_umap(validlabels_id,:);
auxiliary_variables_neuralspace = spike_matrix_animal(validlabels_id,:);
position_labels = spike_matrix_label_pos(validlabels_id);
condition_labels = spike_matrix_label(validlabels_id,2);
trialcount_labels = spike_matrix_label(validlabels_id,3);
time_labels = spike_matrix_label(validlabels_id,4);
epoch_labels = spike_matrix_ep(validlabels_id);
%% Population vector overlap/correlation (PVO)
pos_bins = unique(position_labels);
PVO_all = [];
for i = 1:length(pos_bins)
    currentpos = pos_bins(i);
    currentids = find(position_labels == currentpos);
    current_features = auxiliary_variables_neuralspace(currentids,:);
    if length(currentids) > 1
        pairind = combnk(1:length(currentids),2);
        PVO_tmp = [];
        for pair = 1:length(pairind(:,1))
            cid1 = pairind(pair,1);
            cid2 = pairind(pair,2);
            current_features1 = current_features(cid1,:);
            current_features2 = current_features(cid2,:);
            temp = sum(current_features1.*current_features2,2)./ (sqrt(sum(current_features2.^2,2)) .* sqrt(sum(current_features1.^2,2)));
            PVO_tmp = [PVO_tmp;temp];
        end
        PVO_all =[PVO_all; currentpos, nanmean(PVO_tmp),nanstd(PVO_tmp)]; %[position, mean PVcorr, std PVcorr]
    end
end
%% plot neural manifolds
% labeled by trajectories and positions 
figure,
ep_ids = find(epoch_labels == epochs(1) & trialcount_labels ~=1 & trialcount_labels ~=2 & trialcount_labels ~=35);
scatter3(auxiliary_variables(ep_ids,3),auxiliary_variables(ep_ids,2),auxiliary_variables(ep_ids,1),25*ones(size(position_labels(ep_ids))),position_labels(ep_ids),'filled')
hold on
ep_ids = find(epoch_labels == epochs(2) & trialcount_labels ~=103 &trialcount_labels ~=43);
scatter3(auxiliary_variables(ep_ids,3),auxiliary_variables(ep_ids,2),auxiliary_variables(ep_ids,1),25*ones(size(position_labels(ep_ids))),position_labels(ep_ids)+2,'filled')
map2 = slanCM('PRGn');
traj_colormap=[slanCM('vik');map2(end:-1:1,:)];
colormap(traj_colormap)
view([-34,31])
% labeled by trial time 
traj2 = viridis(alltr_count);
traj_colormap = traj2;
figure,
ep_ids = find(epoch_labels == epochs(2) & trialcount_labels ~=103 &trialcount_labels ~=43);
scatter3(auxiliary_variables(ep_ids,3),auxiliary_variables(ep_ids,2),auxiliary_variables(ep_ids,1),25*ones(size(trialcount_labels(ep_ids))),trialcount_labels(ep_ids),'filled')
hold on
ep_ids = find(epoch_labels == epochs(1) & trialcount_labels ~=1 & trialcount_labels ~=2 & trialcount_labels ~=35);
scatter3(auxiliary_variables(ep_ids,3),auxiliary_variables(ep_ids,2),auxiliary_variables(ep_ids,1),25*ones(size(trialcount_labels(ep_ids))),trialcount_labels(ep_ids),'filled')
colormap(traj_colormap)
view([-34,31])
