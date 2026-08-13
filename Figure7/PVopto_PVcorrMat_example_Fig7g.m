clc
close all
clear all;
%% add codes to path
defaultGraphicsSetttings % graphic settings
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PVR4';
day = 20;
ep = 4;
daystring = num2str(day);
dir = [dir,animalprefix,'\day',daystring,'\'];
prefix = ['day',daystring];
%% define parameters
conditions = 2; % trajectory types, inbound vs. outbound  
pos_binnum = 100; % number of position bins
pos_bin = 1:100; % position bin numbers
%% pre-select the cells with place-field specificity > 1
load(fullfile(dir,  [prefix,'.firingMaps_stats.cellinfo.mat']));% load statistics of firing maps
cell_indx = [];
hpnum = length(firingMaps_stats{ep});
for i = 1:hpnum
    specificity = [];
    stats1 = firingMaps_stats{ep}{i};
    for track = 1:conditions
        pval1 = stats1{track}.specificity;
        specificity = [specificity;pval1];
    end

    if (max(specificity) > 1)
        cell_indx = [cell_indx;i];
    end
end
%% load data
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info
load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load trial-by-trial firing rate maps

% load CA1 PYR spikes
spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell");
cellnum = length(spikes.UID);
%% gather trial-trial population vectors
for track = 1:conditions % inbound and outbound
    tr_count = 0; %reset
    epochtimes = MergePoints.timestamps(ep,:); % epoch start and end time
    trialid_ep = find(behavior.trials(:,1) >= epochtimes(1) &  behavior.trials(:,2) <= epochtimes(2));
    trial_condition = behavior.trialConds(trialid_ep);
    trIDs = find(trial_condition == track);
    %% trial loop
    for tr = 1:length(trIDs)
        tr_count = tr_count +1;
        rm1 = []; % ratemap matrix
        tr_no = trialid_ep(trIDs(tr));
        % cell loop
        for i = 1:length(cell_indx)
            if ismember(firingMaps_trials{cell_indx(i)}.UID, spikes.UID) % valid cells
                linfield1 = firingMaps_trials{cell_indx(i)}.rateMaps;
            else
                linfield1 = [];
            end

            if ~isempty(linfield1)
                linfield1(linfield1 < 0) = nan;% replace -1 with NaN
                linfield_hp = linfield1(tr_no,:);
                a = find(isnan(linfield_hp));
                %pad nan
                if ~isempty(a)
                    [lo,hi]= findcontiguous(a);  %find contiguous NaNs
                    for ii = 1:length(lo)
                        if lo(ii) > 1 & hi(ii) < length(linfield_hp)
                            fill = linspace(linfield_hp(lo(ii)-1), ...
                                linfield_hp(hi(ii)+1), hi(ii)-lo(ii)+1);
                            linfield_hp(lo(ii):hi(ii)) = fill;
                        end
                    end
                end

            else
                linfield_hp = zeros(size(1:pos_binnum));
            end
            [rm1_peak,rm1_peakloc] = max(linfield_hp);
            rm1_peakloc = pos_bin(rm1_peakloc);
            linfield_hp(isnan(linfield_hp)) = 0;
            idx = find(linfield_hp > 0.25*rm1_peak);
            sparsity = length(idx)/length(linfield_hp);
            rm1 = [rm1;linfield_hp];
        end
        rm{track}{tr_count} = rm1; % gather
    end
end
%% calculate PVO (population vector overlap/correlation) matrix
for track = 1:conditions % inbound and outbound
    tr_number = length(rm{track});
    pairind = combnk(1:tr_number,2);
    PVO_temp = ones(tr_number,tr_number);
    for pair = 1:length(pairind(:,1))
        trialpair = pairind(pair,:);
        trialno = trialpair;
        rm1 = rm{track}{trialno(1)};
        rm2 = rm{track}{trialno(2)};
        PVO_temp(trialpair(1),trialpair(2)) = corr(rm1(:),rm2(:),'rows','complete','type','Pearson');
        PVO_temp(trialpair(2),trialpair(1))  = corr(rm1(:),rm2(:),'rows','complete','type','Pearson');
    end
    PVO{track} = PVO_temp;
end
%% plot the average PVO matrix
figure
aa =(PVO{1} +PVO{2})/2;
imagesc(aa([2:3,5:end],[2:3,5:end])) % trials 1 and 4 has some large noises, remove
axis square
caxis([0.4 0.9])
%% calculate PVO confusion matrix for both direaction
tr_number1 = length(rm{1});
tr_number2 = length(rm{2});
tr_number = tr_number1 + tr_number2;
pairind = combnk(1:tr_number,2);
PVO_all = ones(tr_number,tr_number);
for pair = 1:length(pairind(:,1))
    trialpair = pairind(pair,:);
    trialno = trialpair;
    if trialno(1) <= tr_number1
        rm1 = rm{1}{trialno(1)};
    else
        rm1 = rm{2}{trialno(1) - tr_number1};
    end

    if trialno(2) <= tr_number2
        rm2 = rm{1}{trialno(2)};
    else
        rm2 = rm{2}{trialno(2) - tr_number2};
    end
    PVO_all(trialpair(1),trialpair(2)) = corr(rm1(:),rm2(:),'rows','complete','type','Pearson');
    PVO_all(trialpair(2),trialpair(1))  = corr(rm1(:),rm2(:),'rows','complete','type','Pearson');
end
%% Plot the trajectory-specific PVO matrix, showing remapping across two trajectory types
figure
imagesc(PVO_all([2:3,5:tr_number1,tr_number1+1:tr_number1+3,tr_number1+5:end],...
    [2:3,5:tr_number1,tr_number1+1:tr_number1+3,tr_number1+5:end])) % trials 1 and 4 has some large noises, remove
axis square
caxis([0.4 0.9])