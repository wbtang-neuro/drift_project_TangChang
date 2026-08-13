close all
clear all;
clc
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\canonical-correlation-maps-main')) % communication subspace toolbox (Canonical Correlation Analysis,CCA), see Semedo 2022
defaultGraphicsSetttings % graphic settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP20'; % animal prefix
prefix = 'day10'; % day
dir = [dir,animalprefix,'\',prefix,'\'];
eps_RUN = [2,4]; % two behavioral epochs to compare
%% define parameters
speedthresh = 5; % cm/s, running speed threshold
nRun = 100; % number of shuffles
%% load spikes
spike_CA1 = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");
spike_CA3 = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA3");

CA1_num = length(spike_CA1.times);
CA3_num = length(spike_CA3.times);
%% get spike matrix
% combine CA1 and CA3 into a single matrix
% CA1
spikes = [];
for i = 1:length(spike_CA1.times)
    timestamps = spike_CA1.times{i};
    id = ones(size(timestamps))*i; % the id for that unit
    spikes = [spikes; timestamps id]; % add [timestamps id] for this unit to the matrix
end
% attach CA3 after CA1
for i = 1:length(spike_CA3.times)
    timestamps = spike_CA3.times{i};
    id = ones(size(timestamps))*(i + CA1_num); % the id for that unit
    spikes = [spikes; timestamps id]; % add [timestamps id] for this unit to the matrix
end
spikes = sortrows(spikes); % sort spikes accoring to their time
%% load the behavioral file
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
vel  = behavior.speed_smooth; % running speed
% get behavioral labels
linpos = [];% [time, linearized pos.]
for tr = 1:length(behavior.trials(:,1))
    if behavior.trialConds(tr) == 2
        linpos = [linpos;behavior.linpos_trials{tr}(:,1),behavior.linpos_trials{tr}(:,2)+140];
    else
        linpos = [linpos;behavior.linpos_trials{tr}];
    end
end
%% restrict the spike matrix and behavioral labels to running periods
% first epoch
epochtimes = [behavior.epochs{eps_RUN(1)}.startTime, behavior.epochs{eps_RUN(1)}.stopTime];
validid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
vel_ep = vel(validid,:);

RUN = vec2list(vel_ep(:,2) > speedthresh,vel_ep(:,1)); % generate [start end] list of immobile epochs
taskIntervals = SplitIntervals(RUN,'pieceSize',0.05); % 50 ms bins in behavior
[~,spikeMat] = TemplatesCorrelationMat(spikes,'bins',taskIntervals); % CA1CA3 zscored spike matrix
X = spikeMat(:,1:CA1_num);
Y_CA3 = spikeMat(:,1+CA1_num:end);

% restrict behavioral labels to running periods
linpos_RUN1 = [];
for i = 1:length(taskIntervals(:,1))
    timerange = taskIntervals(i,:);
    [pos_diff,idx] = min(abs(linpos(:,1) - timerange(1)));
    if pos_diff < 2
        linpos_RUN1 = [linpos_RUN1; linpos(idx,2)];
    else
        linpos_RUN1 = [linpos_RUN1;nan];
    end
end

% second epoch
epochtimes = [behavior.epochs{eps_RUN(2)}.startTime, behavior.epochs{eps_RUN(2)}.stopTime];
validid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
vel_ep = vel(validid,:);

RUN = vec2list(vel_ep(:,2) > speedthresh,vel_ep(:,1)); % generate [start end] list of immobile epochs
taskIntervals = SplitIntervals(RUN,'pieceSize',0.05); % 50 ms bins in behavior
[~,spikeMat] = TemplatesCorrelationMat(spikes,'bins',taskIntervals); % CA1CA3 zscored spike matrix
X2 = spikeMat(:,1:CA1_num);
Y2_CA3 = spikeMat(:,1+CA1_num:end);

% restrict behavioral labels to running periods
linpos_RUN2 = [];
for i = 1:length(taskIntervals(:,1))
    timerange = taskIntervals(i,:);
    [pos_diff,idx] = min(abs(linpos(:,1) - timerange(1)));
    if pos_diff < 2
        linpos_RUN2 = [linpos_RUN2; linpos(idx,2)];
    else
        linpos_RUN2 = [linpos_RUN2;nan];
    end
end
%% Canonical Correlation Analysis (CCA) with cross-validation
Ntarget = length(Y_CA3(1,:));
Nsource = length(X(1,:));
C_CV_NUM_FOLDS = 10; % 10-fold cross-validation
N = length(Y_CA3(:,1));
c = cvpartition(N, 'kFold', C_CV_NUM_FOLDS);
numPairs = min( size(X, 2), size(Y_CA3, 2) );
cvr = zeros(10, numPairs);
for foldIdx = 1:C_CV_NUM_FOLDS
    cvr(foldIdx,:) = CanonCorrFitAndPredict( ...
        X(c.training(foldIdx),:), ...
        Y_CA3(c.training(foldIdx),:), ...
        X(c.test(foldIdx),:), ...
        Y_CA3(c.test(foldIdx),:) ...
        );
end

cvr = nanmean(cvr);
%% real CCA
[A, B, ~] = canoncorr( X, Y_CA3 );
numPairs = min( size(A, 2), size(B, 2) );

r = zeros(1, numPairs);
for pairIdx = 1:numPairs
    r(pairIdx) = corr( X2*A(:,pairIdx), Y2_CA3*B(:,pairIdx) );
end
r_mean = r;
%% control1: shuffled CCA
for run = 1:nRun
    rng(run, 'twister');
    for i = 1:Ntarget
        shift = randi(N,1);
        Y2_CA3_shuf(:,i) = circshift(Y2_CA3(:,i),shift);
    end
    for i = 1:Nsource
        shift = randi(N,1);
        X2_shuf(:,i) = circshift(X2(:,i),shift);
    end

    for pairIdx = 1:numPairs
        r_tmp(pairIdx) = corr( X2_shuf*A(:,pairIdx), Y2_CA3_shuf*B(:,pairIdx) );
    end
    r_mean_shuf(run,1:numPairs) = r_tmp;
end
%% control2: instead of using CA3 neural activity, use animal's positions for CCA calculation
validid = find(~isnan(linpos_RUN1));
linpos_RUN1 = linpos_RUN1(validid);
Y_CA3_pos = Y_CA3(validid,:);
[A, B, ~] = canoncorr( linpos_RUN1, Y_CA3_pos );
validid = find(~isnan(linpos_RUN2));
linpos_RUN2 = linpos_RUN2(validid);
Y2_CA3_pos = Y2_CA3(validid,:);
r_mean_pos = corr( linpos_RUN2*A(:,1), Y2_CA3_pos*B(:,1) );
%% gather results 
r_all = [max(r_mean),max(prctile(r_mean_shuf,95)),r_mean_pos];%[real, 95th percentile of control 1, control 2]
