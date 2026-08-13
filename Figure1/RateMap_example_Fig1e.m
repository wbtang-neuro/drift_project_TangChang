clc
clear all
close all
%% add base codes to path 
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
defaultGraphicsSetttings % graphics settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP20'; % animal prefix
day = 8; % day
epochs = [2,4]; % behavioral epochs
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% define parameters
speedthresh = 2; % apply speed threshold? cm/s
conditions = 2; % number of trajectory types, 2, left + right
pos_interp = (0:0.005:1)';
% smooth window
nstd = 2;
g1 = gaussian(nstd, 5*nstd);
%% load data
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,'digitalIn.events.mat')) % digitalIn
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info

% load CA1 PYR spikes
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");
%% gather rate maps
trial_conditions = behavior.trialConds; % trajectory types/conditions

% load the trial-by-trial ratemaps
load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load firing rate maps

dCA1_cellnum = length(spikes.UID);
cellnum = length(firingMaps_trials);

% gather rate maps
for epoch = epochs
    epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time
    trialid_ep = find(behavior.trials(:,1) >= epochtimes(1) &  behavior.trials(:,2) <= epochtimes(2));
    trial_conditions_ep = trial_conditions(trialid_ep);
    cond1_trno = trialid_ep(trial_conditions_ep == 1); 
    cond2_trno = trialid_ep(trial_conditions_ep == 2);
    cell_count = 0;
    for i = 1:cellnum
        if ismember(firingMaps_trials{i}.UID, spikes.UID) % dorsal cells
            cell_count = cell_count + 1;
            temp1 = firingMaps_trials{i}.rateMaps(cond1_trno,:);
            temp2 = firingMaps_trials{i}.rateMaps(cond2_trno,:);

            % trajectory type 1
            temp1(isnan(temp1)) = 0;
            temp1(temp1 < 0) = 0;
            temp1 = interp1(0:1/99:1,temp1',pos_interp,'nearest')';
            temp1_sm = [];
            for nn = 1:length(cond1_trno)-2 % average every 3 trials
                tmp = nanmean(temp1(nn:nn+2,:));
                tmp = smoothvect(tmp, g1); % smooth
                temp1_sm = [temp1_sm;tmp];
            end
            [pf_peak1,~] = max(nanmean(temp1_sm));    
            idx = find(nanmean(temp1_sm) > 0.25*pf_peak1);
            sparsity1 = length(idx)/length(nanmean(temp1_sm));

            % trajectory type 2
            temp2(isnan(temp2)) = 0;
            temp2(temp2 < 0) = 0;
            temp2 = interp1(0:1/99:1,temp2',pos_interp,'nearest')';
            temp2_sm = [];
            for nn = 1:length(cond2_trno)-2 % average every 3 trials
                tmp = nanmean(temp2(nn:nn+2,:));
                tmp = smoothvect(tmp, g1); % smooth
                temp2_sm = [temp2_sm;tmp];
            end
            [pf_peak2,~] = max(nanmean(temp2_sm));
            idx = find(nanmean(temp2_sm) > 0.25*pf_peak2);
            sparsity2 = length(idx)/length(nanmean(temp2_sm));

            spike_matrix_trial_cond1(:,:,cell_count) = temp1_sm;
            spike_matrix_trial_cond1_properties(1:2,cell_count) = [pf_peak1,sparsity1];
            spike_matrix_trial_cond2(:,:,cell_count) = temp2_sm;
            spike_matrix_trial_cond2_properties(1:2,cell_count) = [pf_peak2,sparsity2];
        end
    end
    spike_matrix_trial_cond1_all{epoch} = spike_matrix_trial_cond1;
    spike_matrix_trial_cond1_all_properties{epoch} = spike_matrix_trial_cond1_properties;

    spike_matrix_trial_cond2_all{epoch} = spike_matrix_trial_cond2;
    spike_matrix_trial_cond2_all_properties{epoch} = spike_matrix_trial_cond2_properties;

    clear spike_matrix_trial_cond1;
    clear spike_matrix_trial_cond1_properties;

    clear spike_matrix_trial_cond2; 
    clear spike_matrix_trial_cond2_properties;
end
%% select cells with sparsity < 0.5
valid_cellid = find(spike_matrix_trial_cond1_all_properties{epochs(1)}(2,:) < 0.5 |...
    spike_matrix_trial_cond1_all_properties{epochs(2)}(2,:) < 0.5 |...
    spike_matrix_trial_cond2_all_properties{epochs(1)}(2,:) < 0.5 |...
    spike_matrix_trial_cond2_all_properties{epochs(2)}(2,:) < 0.5);
%% all linfields sorted by epoch 1, first trial
% epoch 1, first and last trial
rm1 = [squeeze(nanmean(spike_matrix_trial_cond1_all{2}(1:3,:,valid_cellid)))';squeeze(nanmean(spike_matrix_trial_cond2_all{2}(1:3,:,valid_cellid)))'];
rm2 = [squeeze(nanmean(spike_matrix_trial_cond1_all{2}(end-2:end,:,valid_cellid)))';squeeze(nanmean(spike_matrix_trial_cond2_all{2}(end-2:end,:,valid_cellid)))'];

% epoch 2, first and last trial
rm3 = [squeeze(nanmean(spike_matrix_trial_cond1_all{4}(1:3,:,valid_cellid)))';squeeze(nanmean(spike_matrix_trial_cond2_all{4}(1:3,:,valid_cellid)))'];
rm4 = [squeeze(nanmean(spike_matrix_trial_cond1_all{4}(end-2:end,:,valid_cellid)))';squeeze(nanmean(spike_matrix_trial_cond2_all{4}(end-2:end,:,valid_cellid)))'];

% sort by peak location
[peakrate1,peakid] = max(rm1');
[peakrate2,~] = max(rm2');
[peakrate3,peakid3] = max(rm3');
[peakrate4,~] = max(rm4');
peakrate = max([peakrate1;peakrate2;peakrate3;peakrate4]);

[peakloc_sorted,sortedid] = sort(peakid);
ratemap1 = rm1(sortedid,:)./peakrate(sortedid)';
ratemap2 = rm2(sortedid,:)./peakrate(sortedid)';
ratemap3 = rm3(sortedid,:)./peakrate(sortedid)';
ratemap4 = rm4(sortedid,:)./peakrate(sortedid)';

ratemap1(isnan(ratemap1)) = 0;
ratemap2(isnan(ratemap2)) = 0;
ratemap3(isnan(ratemap3)) = 0;
ratemap4(isnan(ratemap4)) = 0;

zerocellid = find(peakloc_sorted == 0);
allrate = sum(ratemap1(zerocellid,:),2);
[~,indx2]  = sort(allrate);
ratemap2(zerocellid,:) = ratemap2(zerocellid(indx2),:);
ratemap1(zerocellid,:) = ratemap1(zerocellid(indx2),:);
ratemap3(zerocellid,:) = ratemap3(zerocellid(indx2),:);
ratemap4(zerocellid,:) = ratemap4(zerocellid(indx2),:);
%% Plot the sorted rate maps
% define colormap
m=100;
cm_inferno= inferno(m);

figure
subplot(2,4,1)
imagesc(ratemap1)
colormap(cm_inferno)
caxis([0.2 1])
subplot(2,4,2)
imagesc(ratemap2)
colormap(cm_inferno)
caxis([0.2 1])
subplot(2,4,3)
imagesc(ratemap3)
colormap(cm_inferno)
caxis([0.2 1])
subplot(2,4,4)
imagesc(ratemap4)
colormap(cm_inferno)
caxis([0.2 1])
%% all linfields sorted by epoch 2, first trial
[peakloc_sorted,sortedid] = sort(peakid3);
ratemap1 = rm1(sortedid,:)./peakrate(sortedid)';
ratemap2 = rm2(sortedid,:)./peakrate(sortedid)';
ratemap3 = rm3(sortedid,:)./peakrate(sortedid)';
ratemap4 = rm4(sortedid,:)./peakrate(sortedid)';

ratemap1(isnan(ratemap1)) = 0;
ratemap2(isnan(ratemap2)) = 0;
ratemap3(isnan(ratemap3)) = 0;
ratemap4(isnan(ratemap4)) = 0;

zerocellid = find(peakloc_sorted == 0);
allrate = sum(ratemap1(zerocellid,:),2);
[~,indx2]  = sort(allrate);
ratemap2(zerocellid,:) = ratemap2(zerocellid(indx2),:);
ratemap1(zerocellid,:) = ratemap1(zerocellid(indx2),:);
ratemap3(zerocellid,:) = ratemap3(zerocellid(indx2),:);
ratemap4(zerocellid,:) = ratemap4(zerocellid(indx2),:);

subplot(2,4,5)
imagesc(ratemap1)
colormap(cm_inferno)
caxis([0.2 1])
subplot(2,4,6)
imagesc(ratemap2)
colormap(cm_inferno)
caxis([0.2 1])
subplot(2,4,7)
imagesc(ratemap3)
colormap(cm_inferno)
caxis([0.2 1])
subplot(2,4,8)
imagesc(ratemap4)
colormap(cm_inferno)
caxis([0.2 1])
