clc
clear all
close all
%% add base codes to path 
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
defaultGraphicsSetttings % graphics settings
%% list of data, [animal prefix, day, epochs]
animal_info = [{'PPP4'},{10},{[2,4]};...
               {'PPP4'},{11},{[2,4]};...
               {'PPP4'},{14},{[3,5]};...
               {'PPP7'},{8},{[2,4]};...
               {'PPP8'},{7},{[2,4]};...
               {'PPP8'},{8},{[2,4]};...
               {'PPP20'},{4},{[2,4]};...
               {'PPP20'},{5},{[2,4]};...
               {'PPP21'},{5},{[2,4]}
               {'PPP7'},{12},{[3,5]};...
               {'PPP7'},{14},{[2,4]};...
               {'PPP8'},{12},{[2,4]};...
               {'PPP8'},{15},{[2,4,6]};...
               {'PPP8'},{14},{[2,4]};...
               {'PPP8'},{17},{[2,4,6]};...
               {'PPP8'},{19},{[2,6]};...
               {'PPP20'},{7},{[2,4]};...
               {'PPP20'},{8},{[2,4]};...
               {'PPP20'},{13},{[2,4,6]};...
               {'PPP21'},{7},{[2,4]};...
               {'PPP21'},{8},{[2,4]};...
               {'PPP21'},{9},{[2,4]}];
%% set parameters
conditions = 2; % number of trajectory types, 2, left + right
savedata = 1;
pos_interp = (0:0.005:1)';
pos_include = 31:170; % remove the first and last 15% of the trace to avoid accupancy issues for some animals

%smooth parameters
nstd = 3;
g1 = gaussian(nstd, 8*nstd);
%% gather results
Timecorr_all = []; % reset
for session_list = 1:length(animal_info)
    animalprefix = animal_info{session_list,1};
    day = animal_info{session_list,2};
    epochs = animal_info{session_list,3};
    daystring = num2str(day);
    
    dir = 'W:\data\PPP\'; % base folder

    dir = [dir,animalprefix,'\day',daystring,'\']; % data folder
    prefix = ['day',daystring];
    disp([animalprefix,' Day-',num2str(day)]) % show progress

    % load session info
    load(fullfile(dir,[prefix,'.animal.behavior.mat'])) % behavior
    load(fullfile(dir,[prefix,'.MergePoints.events.mat'])) % Session info
    
    % load CA1 PYR spikes
    if  strcmp(animalprefix,'PPP7')
        spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"dCA1");
    elseif strcmp(animalprefix,'PPP4') || (strcmp(animalprefix,'PPP8') && day == 8) ||...
            (strcmp(animalprefix,'PPP8') && (day == 15 || day == 14 || day == 17 || day == 19)) || strcmp(animalprefix,'PVR1') ||...
            strcmp(animalprefix,'OJR56')|| strcmp(animalprefix,'PVR4') || strcmp(animalprefix,'PVR5')
        spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell");
    else
        spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");
    end
    %% epoch loop
    trial_conditions = behavior.trialConds; % trajectory types/conditions
    
    %load the trial-by-trial ratemaps
    load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load firing rate maps
    
    dCA1_cellnum = length(spikes.UID);
    cellnum = length(firingMaps_trials);
    
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
                trialnum1_label = [];
                for nn = 1:length(cond1_trno)-2 % average every 3 trials
                    temp1_sm = [temp1_sm;nanmean(temp1(nn:nn+2,:))];
                    trialnum1_label = [trialnum1_label;nanmean(cond1_trno(nn:nn+2))];
                end
                [pf_peak1,~] = max(nanmean(temp1_sm));    
                idx = find(nanmean(temp1_sm) > 0.25*pf_peak1);
                sparsity1 = length(idx)/length(nanmean(temp1_sm));
    
                % trajectory type 2
                temp2(isnan(temp2)) = 0;
                temp2(temp2 < 0) = 0;
                temp2 = interp1(0:1/99:1,temp2',pos_interp,'nearest')';
                temp2_sm = [];
                trialnum2_label = [];
                for nn = 1:length(cond2_trno)-2 % average every 3 trials
                    temp2_sm = [temp2_sm;nanmean(temp2(nn:nn+2,:))];
                    trialnum2_label = [trialnum2_label;nanmean(cond2_trno(nn:nn+2))];
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
        % select cells with peak rate > 3 Hz and sparsity < 0.5
        valid_cellid = find((spike_matrix_trial_cond1_properties(1,:) > 3 &...
           spike_matrix_trial_cond1_properties(2,:) < 0.5) |...
            (spike_matrix_trial_cond2_properties(1,:) > 3 &...
           spike_matrix_trial_cond2_properties(2,:) < 0.5)) ;

        npos = length(spike_matrix_trial_cond1(1,:,1));
        u1 = spike_matrix_trial_cond1(:,:,valid_cellid);
        u2 = spike_matrix_trial_cond2(:,:,valid_cellid);

        clear spike_matrix_trial_cond1; clear spike_matrix_trial_cond2
        clear spike_matrix_trial_cond2_properties; clear spike_matrix_trial_cond1_properties
        %% Calculate single-cell correlation
        % cell loop
        for cell = 1:length(valid_cellid)
            % calculate the neural distance for each pair of trials
            for t = pos_include % exclude the first and last 15% of the track
                % trajectory type 1
                temp = squeeze(u1(:,t,cell));
                du1 = temp;

                rdv1 = pdist(du1,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
                rdm1 = squareform(rdv1);% this is the distance matrix
                rdm1(isnan(rdm1)) = 0;
                if t == pos_include(1)
                    rdm1_avg = rdm1;
                else
                    rdm1_avg = rdm1_avg + rdm1;% average over all position bins
                end

                % trajectory type 1
                temp = squeeze(u2(:,t,cell));
                du2 = temp;

                rdv2 = pdist(du2,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
                rdm2 = squareform(rdv2);% this is the distance matrix
                rdm2(isnan(rdm2)) = 0;
                if t == pos_include(1)
                    rdm2_avg = rdm2;
                else
                    rdm2_avg = rdm2_avg + rdm2;% average over all position bins
                end

            end
            rdm1_avg = rdm1_avg./length(pos_include);
            rdm2_avg = rdm2_avg./length(pos_include);

            rdmidx = tril(ones(length(trialnum1_label),length(trialnum1_label)),-1);% take the triangular part of matrix
            trilidx = find(rdmidx);
            rdm1_tril = rdm1_avg(trilidx);

            % calculate the trial distance
            rdv1 = pdist(trialnum1_label,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
            rdm_trialcount1 = squareform(rdv1);% this is the distance matrix
            rdm_trialcount1_tril = rdm_trialcount1(trilidx);

            rdmidx = tril(ones(length(trialnum2_label),length(trialnum2_label)),-1);
            trilidx = find(rdmidx);
            rdm2_tril = rdm2_avg(trilidx);

            rdv2 = pdist(trialnum2_label,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
            rdm_trialcount2 = squareform(rdv2);% this is the distance matrix
            rdm_trialcount2_tril = rdm_trialcount2(trilidx);

            % calculate the neural-trial correlation
            r1_n_population = corr(rdm1_tril,rdm_trialcount1_tril,'type','spearman');% neural-tr
            r2_n_population = corr(rdm2_tril,rdm_trialcount2_tril,'type','spearman');% neural-tr

            % gather the correlation
            Timecorr_all = [Timecorr_all;r1_n_population;r2_n_population];
        end
    end
end

