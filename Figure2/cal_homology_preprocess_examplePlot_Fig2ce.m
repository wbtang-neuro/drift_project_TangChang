% this script demonstrates the preprocessing for homology calculation, and the examples of
% the M1 (familiar) manifold (set epoch = 3) with one loop,
% and M2 (novel) manifold (set epoch = 5) with two loops in Fig. 2c (left panels).
clc
clear all
close all
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\slanCM\slanCM\')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\umapAndEppFileExchange (4.1)')) % UMAP toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\codetyt-PersistentHomologyOnMATLAB')) %(co)homology toolbox
defaultGraphicsSetttings % graphic settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP7'; % animal prefix
day = 12; % dat
epoch = 3; % epoch, 3 for M1-familiar, 5 for M2-novel
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% set parameters
speedthresh = 2; % apply speed threshold? cm/s
cellthresh = 5; % minimal number of cells active in a given bin
nstd = 1;
g1 = gaussian(nstd, 3*nstd);
bin = 0.2; % 200 ms
nbins = 100; % downsample to 100 points for robust homology calculation
savefile = 0; % save result?
%% load data
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior

load(fullfile(dir,'digitalIn.events.mat')) % digitalIn
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info
epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time

% load CA1 PYR spikes
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"dCA1");
%% get running periods
% get labels for each bin
linpos = [];% time, linearized pos, trial no.
for tr = 1:length(behavior.trials(:,1))
    if behavior.trialcorrect(tr) == 1
        if behavior.trialConds(tr) == 2
            linpos = [linpos;behavior.linpos_trials{tr}(:,1),behavior.linpos_trials{tr}(:,2)+140,tr*ones(size(behavior.linpos_trials{tr}(:,1)))]; % [time, linearized pos, trial no.]
        else
            linpos = [linpos;behavior.linpos_trials{tr},tr*ones(size(behavior.linpos_trials{tr}(:,1)))];% [time, linearized pos, trial no.]
        end
    end
end
vel = behavior.speed_smooth;% time,running velocity
% take one epoch
epid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
vel = vel(epid,:);
trial_time = list2vec(behavior.trials,vel(:,1));
RUN = vec2list((vel(:,2) > speedthresh) & trial_time,vel(:,1)); % generate [start end] list of running periods
taskIntervals = SplitIntervals(RUN,'pieceSize',bin); % 200 ms bins in behavior
%% restrict position and trial info to running period only
linpos_RUN = [];
for i = 1:length(taskIntervals(:,1))
    timerange = taskIntervals(i,:);
    idx = find(linpos(:,1) >= timerange(1) & linpos(:,1) <= timerange(2));
    if ~isempty(idx)
        linpos_RUN = [linpos_RUN; nanmean(linpos(idx,2:3))];
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
    temp_fit =locsmooth(temp,1/bin,5);
    temp_fit(temp_fit <0)  = 0;
    n(:,unit) = temp_fit;
end
%% Remove bins without enough spikes
validid = [];
for i = 1:length(n(:,1))
    temp = n(i,:);
    id = find(temp > 0);
    if length(id) >= cellthresh && ~isnan(linpos_RUN(i,1))
        validid = [validid;i];
    end
end
spike_matrix_animal = n(validid,:); % spikes
spike_matrix_label = linpos_RUN(validid,2); % labels
%% Run UMAP
% here, I used the previously saved template for reproducibility, but you
% can run it anew, and test the robustness of this manifold, below:
template_filename = [animalprefix,'D',daystring,'EP',num2str(epoch),'_topology.mat'];
% spike_umap = run_umap(spike_matrix_animal,'min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3,'save_template_file',template_filename); %run anew
spike_umap = run_umap(spike_matrix_animal,'min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3,'template_file',template_filename); % use template
%% get the centroid for each position bin for robust estimation of topology
maxpos = ceil(max(linpos_RUN(:,1)));
minpos = floor(min(linpos_RUN(:,1)));
posbin = minpos:(maxpos-minpos)/nbins:maxpos;
[~,ind] = histc(linpos_RUN(validid,1),posbin);

spike_umap_avg = nan(nbins, length(spike_umap(1,:)));
for i = 1:nbins
    currentid = find(ind == i);
    temp = nanmedian(spike_umap(currentid,:));
    spike_umap_avg(i,:) = temp;
end

% remove NaN
temp = sum(spike_umap_avg,2);
nonNaN = find(~isnan(temp));
spike_umap_avg = spike_umap_avg(nonNaN,:);
spike_matrix_label = posbin(nonNaN+1);
%% plot the point cloud
aa = find(spike_matrix_label > 0); % valid bins
map2 = slanCM('vik');
traj_colormap=[map2(end:-1:1,:)];
figure,scatter(spike_umap_avg(aa,1),spike_umap_avg(aa,2),60*ones(size(spike_matrix_label(aa))),spike_matrix_label(aa),'filled')
colormap(traj_colormap)
axis square
xlim([-8,8])
ylim([-10,10])
%% calculate persistence diagram (PD)
[PD, Rinfs] = get_PD_H01_from_2Ddata(spike_umap_avg(aa,[1,2]));
PD_length = PD{2}(:,2) - PD{2}(:,1);
[PD_sorted,~] = sort(PD_length,'descend');
PD_ratio = PD_sorted(1)/(PD_sorted(1)+PD_sorted(2));
% Visualization
VisualizePersistentDiagram(PD, Rinfs);
%% save data?
if savefile
    filename = [animalprefix,prefix,'_EP',num2str(epoch),'_topology_result.mat'];
    save(filename)
end
