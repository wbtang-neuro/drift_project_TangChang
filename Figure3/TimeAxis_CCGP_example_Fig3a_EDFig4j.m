clc
clear all
close all
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
defaultGraphicsSetttings % graphics settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP8'; % animal prefix
day = 17; % day 
epochs = [2,4,6]; % behavioral epochs
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% set parameters
tr_cond = 2; % number of trajectory types, 2, left + right
pos_interp = (0:0.005:1)';
%% load data
% load files
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info

% load CA1 PYR spikes
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");

%load the trial-by-trial ratemaps
load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load firing rate maps
%% gather rate maps
trial_conditions = behavior.trialConds; % trajectory types/conditions

spike_matrix_animal = [];% reset
spike_matrix_label = [];
alltr_count = 0;
dCA1_cellnum = length(spikes.UID);
cellnum = length(firingMaps_trials);
ratemap_var_all = [];
for i = 1:cellnum
    ratemap_var_cell = [];
    trial_count = [];
    epoch_count = [];
    for epoch = epochs
        epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time
        trialid_ep = find(behavior.trials(:,1) >= epochtimes(1) &  behavior.trials(:,2) <= epochtimes(2));
        trial_correct  = behavior.trialcorrect(trialid_ep);
        trialtime = mean(behavior.trials(trialid_ep,:),2);
        trial_condition = behavior.trialConds(trialid_ep);

        alltr_cond = find(trial_condition == tr_cond & trial_correct);% take only correct trials

        tr_no = trialid_ep(alltr_cond);
        trial_count = [trial_count;tr_no];
        epoch_count = [epoch_count;epoch*ones(size(tr_no))];
        temp1_avg = nanmean(firingMaps_trials{i}.rateMaps(tr_no,:)); % averaged rate map
        temp1_avg(isnan(temp1_avg)) = 0;
        linfield_hp_avg = interp1(0:1/99:1,temp1_avg,pos_interp,'nearest');
        for tr_count = 1:length(tr_no)
            tr = tr_no(tr_count);
            temp1 = firingMaps_trials{i}.rateMaps(tr,:);
            temp1(isnan(temp1)) = 0;
            linfield_hp = interp1(0:1/99:1,temp1,pos_interp,'nearest');

            linfield_hp = linfield_hp; %remove from the mean
            ratemap_var_cell = [ratemap_var_cell,linfield_hp];
        end
    end
    ratemap_var_all = [ratemap_var_all;ratemap_var_cell];

end
PVcorr = corr(ratemap_var_all);
%% MDS
dm = 1 - PVcorr; % dissimilarity
[mdssim stressValue] = mdscale(dm,3,'criterion','metricstress','start','cmdscale');
mdssim = mdssim(:,[1,3]);
%% plot MDS results
% get trials for each epoch
epoch1_trno = find(epoch_count == epochs(1));
epoch2_trno = find(epoch_count == epochs(2));
epoch3_trno = find(epoch_count == epochs(3));
% define colormap
traj_colormap = [summer(length(epoch1_trno));plasma(length(epoch2_trno));abyss(length(epoch3_trno)-1)];
% if you have a MATLAB version older than 2023b
% traj_colormap = [summer(length(epoch1_trno));plasma(length(epoch2_trno));gray(length(epoch3_trno)-1)];


% dot sizes for each trial
size_vec = [(3:-(3-0.25)/(length(epoch1_trno)-1):0.25)'; (3:-(3-0.25)/(length(epoch2_trno)-1):0.25)';(3:-(3-0.25)/(length(epoch3_trno)-1):0.25)'];

mdssim1 = mdssim(epoch1_trno,:);
mdssim2 = mdssim(epoch2_trno,:);
mdssim3 = mdssim(epoch3_trno,:);

figure
hold on
plot(mdssim1(:,1),mdssim1(:,2),'Color',traj_colormap(epoch1_trno(5),:),'LineWidth',1)
plot(mdssim2(:,1),mdssim2(:,2),'Color',traj_colormap(epoch2_trno(5),:),'LineWidth',1)
plot(mdssim3(:,1),mdssim3(:,2),'Color',traj_colormap(epoch3_trno(5),:),'LineWidth',1)
scatter(mdssim(:,1),mdssim(:,2),60*size_vec,trial_count,'filled','MarkerFaceAlpha',0.5)
colormap(traj_colormap)
grid on
xlim([-0.8,0.8])
ylim([-0.4,0.6])
axis square 
colorbar
%% get the changing direction with PCA
trnum1 = length(epoch1_trno);
trnum2 = length(epoch2_trno);

training_data = mdssim1;
training_label = epoch1_trno;

dat_all.X = training_data';
dat_all.y = training_label';

test_data = mdssim2;
test_label = epoch2_trno;
%% PCA
X = training_data;
[coeff,score,roots] = pca(X);
basis = coeff(:,1:2);
[n,p] = size(X);
meanX = mean(X,1);
Xfit = repmat(meanX,n,1) + score(:,1:2)*coeff(:,1:2)';

X = training_data;
mu_train = mean(X);

mu = mean(X);

dat_all.X = training_data';
dat_all.y = training_label';

U =  coeff(:,1); % take the first PC
%% project the training data
traj_colormap = [summer(length(epoch1_trno))];

% projecting data points onto the first discriminant axis
Xcentred = bsxfun(@minus, X, mu);
Xprojected = Xcentred * U*transpose(U);
Xprojected = bsxfun(@plus, Xprojected, mu);
% plot discriminant axis
plot(mu(1) + U(1)*[-2 2], mu(2) + U(2)*[-2 2], 'k')
% plot projection lines for each data point
for i=1:size(X,1)
    plot([X(i,1) Xprojected(i,1)], [X(i,2) Xprojected(i,2)], '--','Color',traj_colormap(i,:))
end
% plot projected points
size_vec = (3:-(3-0.25)/(length(epoch1_trno)-1):0.25)';
scatter(Xprojected(:,1), Xprojected(:,2), 60*size_vec,trial_count(epoch1_trno),'x','LineWidth',2)
%% project the testing data
traj_colormap = [plasma(length(epoch2_trno))];

X = test_data;
mu = mean(X);

% projecting data points onto the first discriminant axis
Xcentred = bsxfun(@minus, X, mu);
Xprojected = Xcentred * U*transpose(U);
Xprojected = bsxfun(@plus, Xprojected, mu);
for i=1:size(X,1)
    plot([X(i,1) Xprojected(i,1)- mu(1) + mu_train(1)], [X(i,2) Xprojected(i,2)-mu(2)+ mu_train(2)], '--','Color',traj_colormap(i,:))
end
% plot projected points
size_vec = (3:-(3-0.25)/(length(epoch2_trno)-1):0.25)';
scatter(Xprojected(:,1)- mu(1) + mu_train(1), Xprojected(:,2)-mu(2)+ mu_train(2), 60*size_vec,trial_count(epoch2_trno),'x','LineWidth',2)

