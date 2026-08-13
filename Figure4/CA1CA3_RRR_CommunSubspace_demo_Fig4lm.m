close all
clear all;
clc
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\communication-subspace-master')) % communication subspace toolbox (reduced rank regression, RRR), see Semedo 2020
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
%% restrict the spike matrix to running periods
% get zscored spike matrix for epoch 1
epochtimes = [behavior.epochs{eps_RUN(1)}.startTime, behavior.epochs{eps_RUN(1)}.stopTime];
validid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
vel_ep = vel(validid,:);
RUN = vec2list(vel_ep(:,2) > speedthresh,vel_ep(:,1)); % generate [start end] list of immobile epochs
taskIntervals = SplitIntervals(RUN,'pieceSize',0.05); % 50 ms bins in behavior
[~,spikeMat] = TemplatesCorrelationMat(spikes,'bins',taskIntervals); % CA1CA3 zscored spike matrix
Y_CA1 = spikeMat(:,1:CA1_num);
X = spikeMat(:,1+CA1_num:end);
taskIntervals1 = taskIntervals;

% get zscored spike matrix for epoch 2
epochtimes = [behavior.epochs{eps_RUN(2)}.startTime, behavior.epochs{eps_RUN(2)}.stopTime];
validid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
vel_ep = vel(validid,:);
RUN = vec2list(vel_ep(:,2) > speedthresh,vel_ep(:,1)); % generate [start end] list of immobile epochs
taskIntervals = SplitIntervals(RUN,'pieceSize',0.05); % 50 ms bins in behavior
[~,spikeMat] = TemplatesCorrelationMat(spikes,'bins',taskIntervals); % CA1CA3 zscored spike matrix
Y2_CA1 = spikeMat(:,1:CA1_num);
X2 = spikeMat(:,1+CA1_num:end);
%% Communication subspace with reduced rank regression (RRR), cross-validation to check the dimensions needed
SET_CONSTS

Ntarget = length(Y_CA1(1,:));
Nsource = length(X(1,:));

numDimsUsedForPrediction = 1:Ntarget;
regressMethod = @ReducedRankRegress;
% Number of cross validation folds.
cvNumFolds = 10;
% Initialize default options for cross-validation.
cvOptions = statset('crossval');

cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
	numDimsUsedForPrediction, 'LossMeasure', 'NSE');

cvl = crossval(cvFun, Y_CA1, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

% Stores cross-validation results: mean loss and standard error of the
% mean across folds.
cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

% To compute the optimal dimensionality for the regression model, call
% ModelSelect:
optDimReducedRankRegress = ModelSelect...
	(cvLoss, numDimsUsedForPrediction);

% Plot Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
y = 1-cvLoss(1,:);
e = cvLoss(2,:);
figure
errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
    'MarkerFaceColor', COLOR(V2,:), 'MarkerSize', 10)

xlabel('Number of predictive dimensions')
ylabel('Predictive performance')
xlim([0,30]) % show the first 30 dimensions
%% The number of predictive dimensions will be the smallest number of dimensions for which predictive performance was within one SEM of the peak performance.
[peak_y,peak_d] = max(y); % peak performance
valid_d = find(y >= peak_y - e(peak_d)); % threshold, see Semedo 2020
dim = valid_d(1); % dimensions needed to characterize the communication subspace
hold on
plot([0,30],[peak_y - e(peak_d),peak_y - e(peak_d)],'k--')
%% prediction based on RRR cross different environments
[B1, B1_,V1] = ReducedRankRegress(Y_CA1,X,1:dim); % RRR
[~, Y1hat] = RegressPredict(Y_CA1, X, B1); % prediciton
Y1hat = Y1hat(:,(dim-1)*Ntarget+1:dim*Ntarget); 
[B2, B2_,V2] = ReducedRankRegress(Y2_CA1,X2,1:dim); % RRR
[~, Y2hat] = RegressPredict(Y_CA1, X, B2); % prediciton
Y2hat = Y2hat(:,(dim-1)*Ntarget+1:dim*Ntarget); 
%% control: prediction based on shuffled data
T = length(Y2_CA1(:,1));
for run = 1:nRun
    rng(run, 'twister'); % for randomization
    for i = 1:Ntarget
        shift = randi(T,1);
        Y2_CA1_shuf(:,i) = circshift(Y2_CA1(:,i),shift);
    end
    for i = 1:Nsource
        shift = randi(T,1);
        X2_shuf(:,i) = circshift(X2(:,i),shift);
    end

    % prediction based on shuffled data
    [B2_shuf, B2_shuf_,V2] = ReducedRankRegress(Y2_CA1_shuf, X2(:,randperm(Nsource)),1:dim);
    [~, Y2hat_shuf] = RegressPredict(Y_CA1, X, B2_shuf);
    Y2hat_shuf = Y2hat_shuf(:,(dim-1)*Ntarget+1:dim*Ntarget);
    corr_shuf(run) = corr(Y1hat(:),Y2hat_shuf(:));
end
%% correlation between the signals in the communicaiton subspace across two mazes/epochs
corr_real = corr(Y1hat(:),Y2hat(:)); % real
corr_all = [corr_real,mean(corr_shuf)];% [real, shuffled]
