clc
clear all
close all
%% add codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
%% list of data, [animal,day,epochs]
animal_info = [{'PPP4'},{10},{[2,4]};...
               {'PPP4'},{11},{[2,4]};...
               {'PPP4'},{14},{[3,5]};...
               {'PPP7'},{8},{[2,4]};...
               {'PPP8'},{7},{[2,4]};...
               {'PPP8'},{8},{[2,4]};...
               {'PPP20'},{4},{[2,4]};...
               {'PPP20'},{5},{[2,4]};...
               {'PPP21'},{5},{[2,4]}];
%% set parameters
conditions = 2; % number of trajectory types, 2, left + right
beforeCP = 0; % 1 = position bins before choice point; 0 = all position bins

pos_interp = (0:0.005:1)';
% remove the first and last 15% of the trace to avoid accupancy issues for some animals
if beforeCP
    pos_include = 31:76;% before choice point
else
    pos_include = 31:170; % all positions
end
%% gather results
Trialdist = []; % gather distance between neighorhood trials, reset
for session_list = 1:length(animal_info)
    animalprefix = animal_info{session_list,1};
    day = animal_info{session_list,2};
    epochs = animal_info{session_list,3};
    daystring = num2str(day);

    dir = 'W:\data\PPP\'; % data folder
 
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

    trial_conditions = behavior.trialConds;  % trajectory types/conditions
    trial_correct = behavior.trialcorrect; % correct or error trial
    load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load firing rate maps
    
    cellnum = length(firingMaps_trials);
    % epoch loop
    for epoch = epochs
        epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time
        % restrict trials to the current epoch
        trialid_ep = find(behavior.trials(:,1) >= epochtimes(1) &  behavior.trials(:,2) <= epochtimes(2));
        trial_conditions_ep = trial_conditions(trialid_ep);
        trial_correct_ep = trial_correct(trialid_ep);

        cond1_trno = trialid_ep(trial_conditions_ep == 1); % trajectory type 1
        cond1_correct = trial_correct_ep(trial_conditions_ep == 1);
        cond2_trno = trialid_ep(trial_conditions_ep == 2); % trajectory type 2
        cond2_correct = trial_correct_ep(trial_conditions_ep == 2);

        % gather the firing rate matrix
        cell_count = 0;
        for i = 1:cellnum
            if ismember(firingMaps_trials{i}.UID, spikes.UID) % dorsal cells
                cell_count = cell_count + 1;
                temp1 = firingMaps_trials{i}.rateMaps(cond1_trno,:);
                temp2 = firingMaps_trials{i}.rateMaps(cond2_trno,:);
    
                temp1(isnan(temp1)) = 0;
                temp1(temp1 < 0) = 0;
                temp1 = interp1(0:1/99:1,temp1',pos_interp,'nearest')';
                
                [pf_peak1,~] = max(nanmean(temp1));    
                idx = find(nanmean(temp1) > 0.25*pf_peak1);
                sparsity1 = length(idx)/length(nanmean(temp1));
    
                temp2(isnan(temp2)) = 0;
                temp2(temp2 < 0) = 0;
                temp2 = interp1(0:1/99:1,temp2',pos_interp,'nearest')';
                
                [pf_peak2,~] = max(nanmean(temp2));    
                idx = find(nanmean(temp2) > 0.25*pf_peak2);
                sparsity2 = length(idx)/length(nanmean(temp2));   
    
                spike_matrix_trial_cond1(:,:,cell_count) = temp1;
                spike_matrix_trial_cond1_properties(1:2,cell_count) = [pf_peak1,sparsity1];
                spike_matrix_trial_cond2(:,:,cell_count) = temp2;
                spike_matrix_trial_cond2_properties(1:2,cell_count) = [pf_peak2,sparsity2];
            end
        end

        % select spatial modulated cells (with a peak rate > 3Hz, and sparsity < 0.5)
        valid_cellid = find((spike_matrix_trial_cond1_properties(1,:) > 3 &...
           spike_matrix_trial_cond1_properties(2,:) < 0.5) |...
            (spike_matrix_trial_cond2_properties(1,:) > 3 &...
           spike_matrix_trial_cond2_properties(2,:) < 0.5)) ;

        npos = length(spike_matrix_trial_cond1(1,:,1));
        u1 = spike_matrix_trial_cond1(:,:,valid_cellid); % trajectory type 1
        u2 = spike_matrix_trial_cond2(:,:,valid_cellid); % trajectory type 2

        clear spike_matrix_trial_cond1; clear spike_matrix_trial_cond2
        clear spike_matrix_trial_cond2_properties; clear spike_matrix_trial_cond1_properties
        %% calculate trial distance
        for t = pos_include % only calculate for valid position bins
            temp = squeeze(u1(:,t,:));
            du1 = temp;

            temp = squeeze(u2(:,t,:));
            du2 = temp;
            % trial distance matrix for trajectory type 1
            rdv1 = pdist(du1,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
            rdm1 = squareform(rdv1);% this is the distance matrix
            rdm1(isnan(rdm1)) = 0;
            if t == pos_include(1)
                rdm1_avg = rdm1;
            else
                rdm1_avg = rdm1_avg + rdm1;
            end

            % trial distance matrix for trajectory type 2
            rdv2 = pdist(du2,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
            rdm2 = squareform(rdv2);% this is the distance matrix
            rdm2(isnan(rdm2)) = 0;
            if t == pos_include(1)
                rdm2_avg = rdm2;
            else
                rdm2_avg = rdm2_avg + rdm2;
            end

        end
        % average trial distance matrix
        rdm1_avg = rdm1_avg./length(pos_include);
        rdm2_avg = rdm2_avg./length(pos_include);

        %% for trajectory type 1
        % take the triangular part 
        rdmidx = tril(ones(length(cond1_trno),length(cond1_trno)),-1);
        trilidx = find(rdmidx);
        rdm1_tril = rdm1_avg(trilidx);

        % calculate trial number difference
        rdv1 = pdist(cond1_trno,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
        rdm_trialcount1 = squareform(rdv1);% this is the distance matrix
        rdm_trialcount1_tril = rdm_trialcount1(trilidx);

        % select neighborhood trials
        incorrect_id = find(cond1_correct(2:end) == 0 & cond1_correct(1:end-1) == 1) + 1; % error-correct
        correct_id = find(cond1_correct(2:end) == 1 & cond1_correct(1:end-1) == 1) + 1; % correct-correct

        % gather distance for neighborhood trials
        if ~isempty(incorrect_id) && ~isempty(correct_id) 
            incorrectdist = [];
            for i = 1:length(incorrect_id)
                rdm_trial1_incorrect = rdm_trialcount1(incorrect_id(i)-1,incorrect_id(i));
                rdm1_avg_incorrect = rdm1_avg(incorrect_id(i)-1,incorrect_id(i));
                incorrectdist = [incorrectdist;rdm1_avg_incorrect/rdm_trial1_incorrect]; % normalized by trial number distance
            end
            correctdist = [];
            for i = 1:length(correct_id)
                rdm_trial1_correct = rdm_trialcount1(correct_id(i)-1,correct_id(i));
                rdm1_avg_correct = rdm1_avg(correct_id(i)-1,correct_id(i));
                correctdist = [correctdist;rdm1_avg_correct/rdm_trial1_correct]; % normalized by trial number distance
            end
            Trialdist = [Trialdist; nanmean(incorrectdist),nanmean(correctdist)];
        end

        %% same for trajectory type 2
        rdmidx = tril(ones(length(cond2_trno),length(cond2_trno)),-1);
        trilidx = find(rdmidx);
        rdm2_tril = rdm2_avg(trilidx);

        rdv2 = pdist(cond2_trno,'Euclidean');% this is a dissimilarity vector. a vector containing the pairwise distance between patterns for each stimulus pair.
        rdm_trialcount2 = squareform(rdv2);% this is the distance matrix
        rdm_trialcount2_tril = rdm_trialcount2(trilidx);

        incorrect_id = find(cond2_correct(2:end) == 0 & cond2_correct(1:end-1) == 1) + 1;
        correct_id = find(cond2_correct(2:end) == 1 & cond2_correct(1:end-1) == 1) + 1;

        if ~isempty(incorrect_id) && ~isempty(correct_id)
             incorrectdist = [];
            for i = 1:length(incorrect_id)
                rdm_trial2_incorrect = rdm_trialcount2(incorrect_id(i)-1,incorrect_id(i));
                rdm2_avg_incorrect = rdm2_avg(incorrect_id(i)-1,incorrect_id(i));
                incorrectdist = [incorrectdist;rdm2_avg_incorrect/rdm_trial2_incorrect];
            end
            correctdist = [];
            for i = 1:length(correct_id)
                rdm_trial2_correct = rdm_trialcount2(correct_id(i)-1,correct_id(i));
                rdm2_avg_correct = rdm2_avg(correct_id(i)-1,correct_id(i));
                correctdist = [correctdist;rdm2_avg_correct/rdm_trial2_correct];
            end
            Trialdist = [Trialdist; nanmean(incorrectdist),nanmean(correctdist)];
        end
    end
end

