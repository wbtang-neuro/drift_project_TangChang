clc
clear all
close all
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\slanCM\slanCM\')) % colormap toolbox
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\umapAndEppFileExchange (4.1)')) % UMAP toolbox
defaultGraphicsSetttings % graphic settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP21'; % animal prefix
day = 8; % day
epochs = [2,4]; % two epochs to align
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% define parameters
speedthresh = 2; % apply speed threshold? cm/s
cellthresh = 5; % threshold for number of cell active in a time bin
bin = 0.2; % 200 ms
nbins = 100; % take 100 key position points for robust alignment 
savedata = 0; % save results?
%% load data
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info

% load CA1 PYR spikes
if  strcmp(animalprefix,'PPP7')
    spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"dCA1");
elseif strcmp(animalprefix,'PPP4') || (strcmp(animalprefix,'PPP8') && day == 8) || (strcmp(animalprefix,'PPP8') && day == 15)||...
        (strcmp(animalprefix,'PPP8') && day == 14)||(strcmp(animalprefix,'PPP8') && day == 13)
    spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell");
else
    spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");
end
%% get running periods
% get labels for each time bin
linpos = [];% [time, linearized pos, trial no.]
for tr = 1:length(behavior.trials(:,1))
    if behavior.trialcorrect(tr) == 1
        if behavior.trialConds(tr) == 2
            linpos = [linpos;behavior.linpos_trials{tr}(:,1),behavior.linpos_trials{tr}(:,2)+140,tr*ones(size(behavior.linpos_trials{tr}(:,1)))]; % [time, linearized pos, trial no.]
        else
            linpos = [linpos;behavior.linpos_trials{tr},tr*ones(size(behavior.linpos_trials{tr}(:,1)))];% [time, linearized pos, trial no.]
        end
    end
end
% get running periods
vel = behavior.speed_smooth;% time,running velocity
trial_time = list2vec(behavior.trials,vel(:,1));
RUN = vec2list((vel(:,2) > speedthresh) & trial_time,vel(:,1)); % generate [start end] list of immobile epochs
taskIntervals = SplitIntervals(RUN,'pieceSize',bin); % 200 ms bins in behavior
%% restrict position labels to running periods
linpos_RUN = [];
for i = 1:length(taskIntervals(:,1))
    timerange = taskIntervals(i,:);
    idx = find(linpos(:,1) >= timerange(1) & linpos(:,1) <= timerange(2));
    if ~isempty(idx)
        if linpos(idx,1) <= MergePoints.timestamps(epochs(1),2)
            linpos_RUN = [linpos_RUN; nanmean(linpos(idx,2:3)),1]; % [linpos, trial no, epoch no.]
        else
            linpos_RUN = [linpos_RUN; nanmean(linpos(idx,2:3)),2]; % [linpos, trial no, epoch no.]
        end
    else
        linpos_RUN = [linpos_RUN;nan,nan,nan];
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
%% remove bins that don't have enough spikes
validid = [];
for i = 1:length(n(:,1))
    temp = n(i,:);
    id = find(temp > 0);
    if length(id) >= cellthresh && ~isnan(linpos_RUN(i,1))
        validid = [validid;i];
    end
end

spike_matrix_animal = n(validid,:);
spike_matrix_label = linpos_RUN(validid,2);
%% Run UMAP
% you may encounter an unexpected error from UMAP toolbox with Windows;
% tested with Mac, worked fine
spike_umap = run_umap(spike_matrix_animal,'min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3);
%% get centroid for each position bin for alignment
for ep = 1:2
    epids = find(linpos_RUN(validid,3) == ep);

    linpos_RUN_ep = linpos_RUN(validid(epids),:);
    spike_umap_all{ep} = spike_umap(epids,:);
    
    maxpos = ceil(max(linpos_RUN_ep(:,1)));
    minpos = floor(min(linpos_RUN_ep(:,1)));
    posbin = minpos:(maxpos-minpos)/nbins:maxpos;
    [~,ind] = histc(linpos_RUN_ep(:,1),posbin);

    spike_umap_label{ep} = [linpos_RUN_ep,ind];

    spike_umap_avg_tmp = nan(nbins, length(spike_umap(1,:)));
    for i = 1:nbins
        currentid = find(ind == i);
        currentid = epids(currentid);
        temp = nanmedian(spike_umap(currentid,:));
        spike_umap_avg_tmp(i,:) = temp;
    end
    spike_umap_avg{ep} = spike_umap_avg_tmp;
end

% remove NaN
temp = sum(spike_umap_avg{1},2) .* sum(spike_umap_avg{2},2);
nonNaN = find(~isnan(temp));
spike_umap_avg{1} = spike_umap_avg{1}(nonNaN,:);
spike_umap_avg{2} = spike_umap_avg{2}(nonNaN,:);
spike_avg_label = posbin(nonNaN+1);

% organize into epochs
for ep = 1:2 % two epochs to align
    labels = find(ismember(spike_umap_label{ep}(:,end),nonNaN));
    spike_umap_all{ep} = spike_umap_all{ep}(labels,:);
    spike_umap_label{ep} = spike_umap_label{ep}(labels,:);
end
%% procrustes transformation
[d,Z,transform] = procrustes(spike_umap_avg{1},spike_umap_avg{2},'reflection', false); % procrustes, no reflection
%% alignment, apply the transformation matrix to epoch 2
label_num = length(spike_avg_label);
spike_umap2_transform = zeros(size(spike_umap_all{2}));
for i = 1:label_num
    current_label = nonNaN(i);
    temp_ids = find(spike_umap_label{2}(:,end) == current_label);
    spike_umap2_transform(temp_ids,:) = transform.b*spike_umap_all{2}(temp_ids,:)*transform.T + transform.c(i,:);
end
%% predict positions after alignment, using KNN
auxiliary_variables  = spike_umap_all{1};
position_labels = spike_umap_label{1}(:,end);

auxiliary_variables_test  = spike_umap2_transform;
position_labels_test = spike_umap_label{2}(:,end);

mdl = fitcecoc(auxiliary_variables,position_labels, 'Learners', 'knn');
prediction = predict(mdl, auxiliary_variables_test);
%% calculate prediction accuracy (r-val)
accuracy = corr(position_labels_test,prediction,'type','spearman');
%% calculate the confusion matrix
actual_matrix = zeros(length(nonNaN),length(nonNaN));
decoded_matrix = zeros(length(nonNaN),length(nonNaN));

actualpos = position_labels_test;
decodedpos = prediction;

for i = 1:length(actualpos)
    [~,id1] = min(abs(nonNaN - actualpos(i)));
    [~,id2] = min(abs(nonNaN - decodedpos(i)));
    if ~isempty(id1) &&  ~isempty(id2)
        actual_matrix(id1,:) = actual_matrix(id1,:) + 1;
        decoded_matrix(id1,id2) =  decoded_matrix(id1,id2)+1;
    end
end

confusion_matrix = decoded_matrix./actual_matrix;
confusion_matrix(find(isnan(confusion_matrix))) = 0;
   
% plot confusion matrix
figure,
imagesc(confusion_matrix)
axis square
caxis([0 0.1])
m=100;
colormap(inferno(m))
title('confusion matrix')
%% plot pre-alignment manifolds
figure,
maxpos = max(position_labels_test(50:end));
hold on
scatter3(spike_umap_all{2}(50:end,1),spike_umap_all{2}(50:end,2),spike_umap_all{2}(50:end,3),30*ones(size(position_labels_test(50:end))),position_labels_test(50:end),'filled')
scatter3(auxiliary_variables(:,1),auxiliary_variables(:,2),auxiliary_variables(:,3),30*ones(size(position_labels)),position_labels+maxpos,'filled')

map2 = slanCM('PRGn',round(max(position_labels_test(50:end))));
traj_colormap=[map2(end:-1:1,:);slanCM('vik',round(maxpos))];
colormap(traj_colormap)
xlabel('D1')
ylabel('D2')
zlabel('D3')
view([-15,50])
grid on
title('before alignment')
%% add the centroids for the plot
nbins = 100;
maxpos = ceil(max(position_labels));
minpos = floor(min(position_labels));
posbin = minpos:(maxpos-minpos)/nbins:maxpos;
[~,ind] = histc(position_labels,posbin);
spike_umap_avg = nan(nbins, length(auxiliary_variables(1,:)));
for i = 1:nbins
    currentid = find(ind == i);
    temp = nanmedian(auxiliary_variables(currentid,:));
    spike_umap_avg(i,:) = temp;
end
% remove NaN
temp = sum(spike_umap_avg,2);
nonNaN = find(~isnan(temp));
spike_umap_avg = spike_umap_avg(nonNaN,:);
spike_umap_label = posbin(nonNaN+1);
scatter3(spike_umap_avg(:,1),spike_umap_avg(:,2),-10*ones(size(spike_umap_avg(:,1))),35*ones(size(spike_umap_label)),spike_umap_label+maxpos,'filled')

nbins = 200;
maxpos = ceil(max(position_labels_test));
minpos = floor(min(position_labels_test));
posbin = minpos:(maxpos-minpos)/nbins:maxpos;
[~,ind] = histc(position_labels_test,posbin);
spike_umap_avg = nan(nbins, length(spike_umap_all{2}(1,:)));
for i = 1:nbins
    currentid = find(ind == i);
    temp = nanmedian(spike_umap_all{2}(currentid,:));
    spike_umap_avg(i,:) = temp;
end
% remove NaN
temp = sum(spike_umap_avg,2);
nonNaN = find(~isnan(temp));
spike_umap_avg = spike_umap_avg(nonNaN,:);
spike_umap_label = posbin(nonNaN+1);
scatter3(-14*ones(size(spike_umap_avg(:,1))),spike_umap_avg(:,2),spike_umap_avg(:,3),35*ones(size(spike_umap_label)),spike_umap_label,'filled')
xlim([-14,12])
ylim([-6,8])
zlim([-10,5])
%% plot after-alignment manifolds
figure,
scatter3(auxiliary_variables(:,1),auxiliary_variables(:,2),auxiliary_variables(:,3),30*ones(size(position_labels)),position_labels+maxpos,'filled','MarkerFaceAlpha',0.05)
hold on
scatter3(auxiliary_variables_test(50:end,1),auxiliary_variables_test(50:end,2),auxiliary_variables_test(50:end,3),30*ones(size(position_labels_test(50:end))),position_labels_test(50:end),'filled')
traj_colormap=[map2(end:-1:1,:);slanCM('vik',round(maxpos))];
colormap(traj_colormap)
xlabel('D1')
ylabel('D2')
zlabel('D3')
%% add the centroids for the plot
nbins = 100;
maxpos = ceil(max(position_labels));
minpos = floor(min(position_labels));
posbin = minpos:(maxpos-minpos)/nbins:maxpos;
[~,ind] = histc(position_labels,posbin);
spike_umap_avg = nan(nbins, length(auxiliary_variables(1,:)));
for i = 1:nbins
    currentid = find(ind == i);
    temp = nanmedian(auxiliary_variables(currentid,:));
    spike_umap_avg(i,:) = temp;
end
% remove NaN
temp = sum(spike_umap_avg,2);
nonNaN = find(~isnan(temp));
spike_umap_avg = spike_umap_avg(nonNaN,:);
spike_umap_label = posbin(nonNaN+1);
scatter3(spike_umap_avg(:,1),spike_umap_avg(:,2),-10*ones(size(spike_umap_avg(:,1))),35*ones(size(spike_umap_label)),spike_umap_label+maxpos,'filled','MarkerFaceAlpha',0.5)

nbins = 200;
maxpos = ceil(max(position_labels_test));
minpos = floor(min(position_labels_test));
posbin = minpos:(maxpos-minpos)/nbins:maxpos;
[~,ind] = histc(position_labels_test,posbin);
spike_umap_avg = nan(nbins, length(auxiliary_variables_test(1,:)));
for i = 1:nbins
    currentid = find(ind == i);
    temp = nanmedian(auxiliary_variables_test(currentid,:));
    spike_umap_avg(i,:) = temp;
end
% remove NaN
temp = sum(spike_umap_avg,2);
nonNaN = find(~isnan(temp));
spike_umap_avg = spike_umap_avg(nonNaN,:);
spike_umap_label = posbin(nonNaN+1);
scatter3(spike_umap_avg(:,1),spike_umap_avg(:,2),-10*ones(size(spike_umap_avg(:,1))),35*ones(size(spike_umap_label)),spike_umap_label,'filled')
xlim([-14,12])
ylim([-6,8])
zlim([-10,5])
view([-15,50])
grid on
title('after alignment')
%% save result?
if savedata
    filename = ['Umap_homo_algin_decoding_',animalprefix,'D',num2str(day),'.mat'];
    save(filename,'position_labels_test','prediction','accuracy')
end


