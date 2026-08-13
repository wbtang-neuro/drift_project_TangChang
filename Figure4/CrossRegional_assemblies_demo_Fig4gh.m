% this script demonstrates how to do cross-regional (CA1-CA3) assembly
% detection (Figs. 3g and 3h nd  Fig. S4)
close all
clear all;
clc
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
defaultGraphicsSetttings % graphic settings
%% define data
dir = 'W:\data\PPP\'; % data folder
animalprefix = 'PPP20';% animal prefix
prefix = 'day8';% day
dir = [dir,animalprefix,'\',prefix,'\'];
eps_RUN = [2,4];% behavioral epochs
%% set parameters
speedthresh = 5; % cm/s,running speed threshold
savedata = 0; % save data?
nstd = 4;% 2 to 4 SDs for thresholding assembly members
%% load spikes
spike_CA1 = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1"); %CA1 PYRs
spike_CA3 = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA3"); %CA3 PYRs

CA1_num = length(spike_CA1.times);
CA3_num = length(spike_CA3.times);
%% get spike matrix
% combine CA1 and CA3 into a single matrix
spikes = [];
% CA1 first
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
%% get running periods during the first epoch only to build template
load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
% first epoch only to detect assemblies
epochtimes = [behavior.epochs{eps_RUN(1)}.startTime, behavior.epochs{eps_RUN(1)}.stopTime];
vel  = behavior.speed_smooth; % running speed
validid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
vel_ep = vel(validid,:);

% try only take one epoch
RUN = vec2list(vel_ep(:,2) > speedthresh,vel_ep(:,1)); % generate [start end] list of immobile epochs
taskIntervals = SplitIntervals(RUN,'pieceSize',0.05); % 50 ms bins in behavior
%% calculate CA1-CA3 correlation matrix
correlationMat = TemplatesCorrelationMat(spikes,'bins',taskIntervals); % CA1CA3 - CA1CA3 matrix
correlationMat_cross = correlationMat(1:CA1_num,CA1_num+1:end); % take the rectangular part, which is the CA1-CA3 correlation. Remove CA1-CA1 and CA3-CA3 correlation
%% SVD
[u0,s0,v0] = svd(correlationMat_cross);
[eigenvalues,i] = sort(diag(s0),'descend');
CA1vec = u0(:,i);
CA3vec = v0(:,i);
%% significant eigenvalues
lambdaMax = 0.7/CA3_num; % Everitt and Dunn, 2001, threshold for removing non-significant components
relative_variances = eigenvalues.^2./(sum(eigenvalues.^2)); % relative variances
significant = relative_variances>lambdaMax;

CA1vec = CA1vec(:,significant);
CA3vec = CA3vec(:,significant);
assemble_num = sum(significant);
%% calculate template and detect assembly memebers
for i = 1:assemble_num
    Template{i} = CA1vec(:,i)*CA3vec(:,i)';
    assemble_mean = nanmean(abs(CA1vec(:,i)));
    assemble_std = nanstd(abs(CA1vec(:,i)));
    threshold = assemble_mean + nstd*assemble_std; % van de Ven 2016
    memeber_IDs = find(abs(CA1vec(:,i)) > threshold);
    assemble_members{i} = memeber_IDs;
end
%% get runing periods for both epochs
vel  = behavior.speed_smooth; % running speed

for ep = eps_RUN
    epochtimes = [behavior.epochs{ep}.startTime, behavior.epochs{ep}.stopTime];
    validid = find(vel(:,1) >= epochtimes(1) & vel(:,1) <= epochtimes(2));
    vel_ep = vel(validid,:);
    thetalist{ep} = vec2list(vel_ep(:,2) > speedthresh,vel_ep(:,1)); % generate [start end] list of immobile epochs
end
%% calculate theta covariance for assembly member pairs, shown in Fig. S4
thetacov_all = []; % reset for gathering
thetacov_all_info = [];

for group = 1:assemble_num
    memeber_IDs = assemble_members{group};
    memeber_num = length(memeber_IDs);
    if memeber_num > 1
        pairind = combnk(1:memeber_num,2);
    
        % calculate theta covariance
        for pair = 1:length(pairind(:,1))
            cid1 = memeber_IDs(pairind(pair,1));
            cid2 = memeber_IDs(pairind(pair,2));
            spikes1 = spike_CA1.times{cid1};
            spikes2 = spike_CA1.times{cid2};
            thetacov_pair = [];
            for ep = eps_RUN
                [timebase,theta_crosscov, thetacov,peaktime] = Pairwise_peakcov_fun(spikes1,spikes2,thetalist{ep});
                thetacov_pair = [thetacov_pair,peaktime];
            end
            thetacov_all = [thetacov_all;thetacov_pair];
            thetacov_all_info = [thetacov_all_info;group,cid1,cid2];
        end
    end
end
%% calculate theta covariance of all pairs, shown in Fig. S4
pairind = combnk(1:CA1_num,2);
thetacov_allpairs = [];
% calculate theta covariance
for pair = 1:length(pairind(:,1))
    cid1 = pairind(pair,1);
    cid2 = pairind(pair,2);
    spikes1 = spike_CA1.times{cid1};
    spikes2 = spike_CA1.times{cid2};
    thetacov_pair = [];
    for ep = eps_RUN
        [timebase,theta_crosscov, thetacov,peaktime] = Pairwise_peakcov_fun(spikes1,spikes2,thetalist{ep});
        thetacov_pair = [thetacov_pair,peaktime];

    end
    thetacov_allpairs = [thetacov_allpairs;thetacov_pair];
end
%% correlation
corr(thetacov_allpairs(:,1),thetacov_allpairs(:,2),'rows','complete')
corr(thetacov_all(:,1),thetacov_all(:,2),'rows','complete')
%% generate raw 2D histogram
% edges for theta covariance is +/- 200 ms, so here we took +/- 190 ms to
% prevent edge effects.
figure
validid = (abs(thetacov_allpairs(:,1)) < 0.195 & abs(thetacov_allpairs(:,2)) < 0.195);
binedges = -0.19:0.02:0.19;
h = histogram2(thetacov_allpairs(validid,1),thetacov_allpairs(validid,2),binedges,binedges,'DisplayStyle','tile','ShowEmptyBins','on','Normalization','probability');
clim([0,0.005])
hold on
validid = (abs(thetacov_all(:,1)) < 0.195 & abs(thetacov_all(:,2)) < 0.195);
plot(thetacov_all(validid,1),thetacov_all(validid,2),'ro')
axis square
%% generate smoothed 2D histogram
validid = (abs(thetacov_allpairs(:,1)) < 0.195 & abs(thetacov_allpairs(:,2)) < 0.195);
hist_prob = h.Values;

binedges_re = -0.19:0.005:0.19;
hist_prob_re = imresize(hist_prob,[77,77])';
hist_sm = imgaussfilt(hist_prob_re,3.5);
figure
imagesc(binedges_re,binedges_re,hist_sm)
clim([0,0.005])
axis square
hold on
validid = (abs(thetacov_all(:,1)) < 0.195 & abs(thetacov_all(:,2)) < 0.195);
plot(thetacov_all(validid,1),thetacov_all(validid,2),'ro')
%% calculate place-field COM shift for cell pairs
% load place fields or rate maps
load(fullfile(dir,  [prefix,'.firingMapsAvg_linear.cellinfo.mat']));% load firing rate maps

COMshift_all = []; % for gathering, reset
COMshift_all_info = [];
posvec = (1:200);

% COM shift for assembly member pairs
for group = 1:assemble_num
    memeber_IDs = assemble_members{group};
    memeber_num = length(memeber_IDs);
    if memeber_num > 1
        pairind = combnk(1:memeber_num,2);

        for pair = 1:length(pairind(:,1))
            cid1 = memeber_IDs(pairind(pair,1));
            cid2 = memeber_IDs(pairind(pair,2));
            UID1 = spike_CA1.UID(cid1);
            UID2 = spike_CA1.UID(cid2);

            COMshift_pair = [];
            for ep = eps_RUN
                ID1 = find(firingMaps_linear{ep}.UID == UID1);
                % concatenate two trajectories
                CA1rate1 = [firingMaps_linear{ep}.rateMaps{ID1}{1}, firingMaps_linear{ep}.rateMaps{ID1}{2}(end:-1:1)];
                CA1rate1(CA1rate1 < 0) = 0;
                ID2 = find(firingMaps_linear{ep}.UID == UID2);
                CA1rate2 = [firingMaps_linear{ep}.rateMaps{ID2}{1}, firingMaps_linear{ep}.rateMaps{ID2}{2}(end:-1:1)];
                CA1rate2(CA1rate2 < 0) = 0;

                COM1 = sum(CA1rate1.*posvec)/sum(CA1rate1);
                COM2 = sum(CA1rate2.*posvec)/sum(CA1rate2);
                [dCOM1,idx1] = min([COM1,200-COM1]);
                [dCOM2,idx2] = min([COM2,200-COM2]);
                d2 = min(COM1,200-COM1)+ min(COM2,200-COM2);
                d1 = abs(COM1 - COM2);
                [dist,type] = min([d1,d2]);
                if type == 1
                    COMshift_pair = [COMshift_pair,(COM1 - COM2)];
                else
                    if idx1 == 1
                        sign = 1;
                    else
                        sign = -1;
                    end
                    COMshift_pair = [COMshift_pair,d2 * sign];
                end
            end
            COMshift_all = [COMshift_all;COMshift_pair/200*2]; % normalize to a +/- 1 range
            COMshift_all_info = [COMshift_all_info;group,cid1,cid2]; % save cell pair info
        end
    end
end
%% COM shift of all pairs
pairind = combnk(1:CA1_num,2);
COMshift_allpairs = [];
% calculate theta cov
for pair = 1:length(pairind(:,1))
    cid1 = pairind(pair,1);
    cid2 = pairind(pair,2);
    UID1 = spike_CA1.UID(cid1);
    UID2 = spike_CA1.UID(cid2);

    COMshift_pair = [];
    for ep = eps_RUN
        ID1 = find(firingMaps_linear{ep}.UID == UID1);
        % concatenate two trajectories
        CA1rate1 = [firingMaps_linear{ep}.rateMaps{ID1}{1}, firingMaps_linear{ep}.rateMaps{ID1}{2}(end:-1:1)];
        CA1rate1(CA1rate1 < 0) = 0;
        ID2 = find(firingMaps_linear{ep}.UID == UID2);
        CA1rate2 = [firingMaps_linear{ep}.rateMaps{ID2}{1}, firingMaps_linear{ep}.rateMaps{ID2}{2}(end:-1:1)];
        CA1rate2(CA1rate2 < 0) = 0;

        COM1 = sum(CA1rate1.*posvec)/sum(CA1rate1);
        COM2 = sum(CA1rate2.*posvec)/sum(CA1rate2);
        [dCOM1,idx1] = min([COM1,200-COM1]);
        [dCOM2,idx2] = min([COM2,200-COM2]);
        d2 = min(COM1,200-COM1)+ min(COM2,200-COM2);
        d1 = abs(COM1 - COM2);
        [dist,type] = min([d1,d2]);
        if type == 1
            COMshift_pair = [COMshift_pair,(COM1 - COM2)];
        else
            if idx1 == 1
                sign = 1;
            else
                sign = -1;
            end
            COMshift_pair = [COMshift_pair,d2 * sign];
        end
    end
    COMshift_allpairs = [COMshift_allpairs;COMshift_pair/200*2];% normalize to a +/- 1 range
end
%% rotate the matrix to align the diagonal 
rval = corr(COMshift_all(:,1),COMshift_all(:,2),'rows','complete');
if rval < 0
    COMshift_all(:,2) = -COMshift_all(:,2);
    COMshift_allpairs(:,2) = -COMshift_allpairs(:,2);
end
%% COM shift scatter plot
figure
hold on
plot(COMshift_allpairs(:,1),COMshift_allpairs(:,2),'k.')
plot(COMshift_all(:,1),COMshift_all(:,2),'ro')
axis square
%% save data
if savedata
    save(fullfile(dir, [basenameFromBasepath(dir),'.COMshift_allpairs.mat']),'COMshift_allpairs','COMshift_all');
end
