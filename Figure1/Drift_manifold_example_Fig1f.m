clc
clear all
close all
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\matplotlib')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\slanCM\slanCM\')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\umapAndEppFileExchange (4.1)')) % UMAP toolbox
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP7'; % animal prefix
day = 8; % day
epochs = [2,4]; % behavioral epochs
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% define parameters
speedthresh = 2; % apply speed threshold? cm/s
cellthresh = 5; % threshold for number of cell active in a time bin
bin = 0.2; % 200 ms
%% load data
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,'digitalIn.events.mat')) % digitalIn
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info

% load CA1 PYR spikes
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"dCA1");
%% get running periods
linpos = [];% time, linearized pos, trial no.
for tr = 1:length(behavior.trials(:,1))
    if behavior.trialcorrect(tr) == 1 % take only correct trials
        if behavior.trialConds(tr) == 2 % concatenate two types of trajectories
            linpos = [linpos;behavior.linpos_trials{tr}(:,1),behavior.linpos_trials{tr}(:,2)+140,tr*ones(size(behavior.linpos_trials{tr}(:,1)))];
        else
            linpos = [linpos;behavior.linpos_trials{tr},tr*ones(size(behavior.linpos_trials{tr}(:,1)))];
        end
    end
end

vel = behavior.speed_smooth;% time,running velocity
trial_time = list2vec(behavior.trials,vel(:,1)); % take only trial time
RUN = vec2list((vel(:,2) > speedthresh) & trial_time,vel(:,1)); % generate [start end] list of running periods
taskIntervals = SplitIntervals(RUN,'pieceSize',bin); % 200 ms bins in behavior
%% restrict position and trial info to running period only
linpos_RUN = [];
for i = 1:length(taskIntervals(:,1))
    timerange = taskIntervals(i,:);
    idx = find(linpos(:,1) >= timerange(1) & linpos(:,1) <= timerange(2));
    if ~isempty(idx)
        linpos_RUN = [linpos_RUN; nanmean(linpos(idx,2:3))]; % [linearized pos, trial no.]
    else
        linpos_RUN = [linpos_RUN;nan,nan];
    end
end
%% get spike matrix
dCA1_cellnum = length(spikes.UID);
spikes_CA1 = [];
for i = 1:dCA1_cellnum
    timestamps = spikes.times{i};
    id = ones(size(timestamps))*i; % the id for that unit
    spikes_CA1 = [spikes_CA1; timestamps id]; % add [timestamps id] for this unit to the matrix
end
spikes_CA1 = sortrows(spikes_CA1,1); % sort spikes accoring to their time 
id = spikes_CA1(:,2);

% Shift spike times to start at 0, and list bins unless explicitly provided
m = min([min(spikes_CA1(:,1)) min(taskIntervals(:))]);
spikes_CA1(:,1) = spikes_CA1(:,1) - m;
taskIntervals = taskIntervals - m;

% Create spike count matrix
nBins = size(taskIntervals,1);
if isempty(nBins), return; end
n = zeros(nBins,dCA1_cellnum);
for unit = 1:dCA1_cellnum
    temp = CountInIntervals(spikes_CA1(id==unit,1),taskIntervals);
    temp_fit =locsmooth(temp,1/bin,5); % local smooth
    temp_fit(temp_fit <0)  = 0;
    n(:,unit) = temp_fit;
end
%% Remove bins without enough spikes
validid = [];
for i = 1:length(n(:,1))
    temp = n(i,:);
    id = find(temp > 0);
    if length(id) >= cellthresh && ~isnan(linpos_RUN(i,1)) % bins 
        validid = [validid;i];
    end
end

spike_matrix_animal = n(validid,:);
spike_matrix_label = linpos_RUN(validid,2);
%% Run UMAP
spike_umap = run_umap(spike_matrix_animal,'min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3,'template_file','PPP7D8_timedecode2.mat');% use a template for reproducibility
%% Plot the manifold example, time labels
spike_matrix_label = linpos_RUN(validid,2); % use trial numbers as labels
aa = find(spike_matrix_label~=42 & spike_matrix_label~=43); % remove two trials with large noise
traj_colormap = viridis(100);
figure,scatter3(spike_umap(aa,1),spike_umap(aa,2),spike_umap(aa,3),27*ones(size(spike_matrix_label(aa))),spike_matrix_label(aa),'filled')
colormap(traj_colormap)
zlim([-8,10])
view([-69,-12])
%% Plot the manifold example, position labels
ep1_trnum = find(behavior.trials(:,2) < MergePoints.timestamps(epochs(1),2));
map2 = slanCM('PRGn');
traj_colormap = [slanCM('vik');map2(end:-1:1,:)];% position colormap
% neural manifold for the first epoch 
aa = find(spike_matrix_label <= ep1_trnum(end) & spike_matrix_label~=42 & spike_matrix_label~=43); % trials from epoch 1
spike_matrix_label_choice = linpos_RUN(validid,1); % use positions as labels
spike_umap1 = spike_umap(aa,:);
spike_umap1_label = spike_matrix_label_choice(aa);
figure,scatter3(spike_umap(aa,1),spike_umap(aa,2),spike_umap(aa,3),27*ones(size(spike_matrix_label_choice(aa))),spike_matrix_label_choice(aa),'filled')
maxpos = max(spike_matrix_label_choice(aa));
hold on

% neural manifold for the second epoch 
aa = find(spike_matrix_label > ep1_trnum(end) & spike_matrix_label~=42 & spike_matrix_label~=43); % trials from epoch 2
scatter3(spike_umap(aa,1),spike_umap(aa,2),spike_umap(aa,3),27*ones(size(spike_matrix_label_choice(aa))),spike_matrix_label_choice(aa)+maxpos,'filled')
spike_umap2 = spike_umap(aa,:);
spike_umap2_label = spike_matrix_label_choice(aa) + maxpos;
view([-69,-12])
zlim([-8,10])
colormap(traj_colormap)
%% Plot the centroids of each position bin
nbins = 400; % bin number
spike_umap_tmp = [spike_umap1;spike_umap2]; % concatenate two manifolds
spike_umap_label = [spike_umap1_label;spike_umap2_label]; % manifold labels -- positions
maxpos = ceil(max(spike_umap_label));
minpos = floor(min(spike_umap_label));
posbin = minpos:(maxpos-minpos)/nbins:maxpos;
[~,ind] = histc(spike_umap_label,posbin);
spike_umap_avg = nan(nbins, length(spike_umap_tmp(1,:)));
% get centroids
for i = 1:nbins
    currentid = find(ind == i);
    temp = nanmedian(spike_umap_tmp(currentid,:));
    spike_umap_avg(i,:) = temp;
end
% remove NaN
temp = sum(spike_umap_avg,2);
nonNaN = find(~isnan(temp));
spike_umap_avg = spike_umap_avg(nonNaN,:);
spike_umap_label = posbin(nonNaN+1);
% plot centroids
scatter3(spike_umap_avg(:,1),spike_umap_avg(:,2),10*ones(size(spike_umap_avg(:,3))),35*ones(size(spike_umap_label)),spike_umap_label,'filled')
