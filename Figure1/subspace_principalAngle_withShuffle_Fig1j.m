clc
clear all
close all
%% add base codes to path 
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
defaultGraphicsSetttings % graphics settings
%% list of data
drift_remap = 1; % 1: run on the drift (M1-M1') data; 0: run on the remap (M1-M2) data
if drift_remap
    % drift sessions
    animal_info = [{'PPP4'},{10},{[2,4]};...
        {'PPP4'},{11},{[2,4]};...
        {'PPP4'},{14},{[3,5]};...
        {'PPP7'},{8},{[2,4]};...
        {'PPP8'},{7},{[2,4]};...
        {'PPP8'},{8},{[2,4]};...
        {'PPP20'},{4},{[2,4]};...
        {'PPP20'},{5},{[2,4]};...
        {'PPP21'},{5},{[2,4]};...
        {'PPP23'},{7},{[2,4]}];
else
    % remap sessions
    animal_info = [{'PPP7'},{12},{[3,5]};...
        {'PPP7'},{14},{[2,4]};...
        {'PPP8'},{12},{[2,4]};...
        {'PPP8'},{14},{[2,4]};...
        {'PPP8'},{19},{[2,6]};...
        {'PPP20'},{7},{[2,4]};...
        {'PPP20'},{8},{[2,4]};...
        {'PPP21'},{7},{[2,4]};...
        {'PPP21'},{8},{[2,4]};...
        {'PPP21'},{9},{[2,4]};...
        {'PPP23'},{8},{[2,4]};...
        {'PPP23'},{9},{[2,4]}];
end
%% session loop
PA_gather = [];% reset, gather the principal angles
for session_list = 1:length(animal_info)
    animalprefix = animal_info{session_list,1};
    day = animal_info{session_list,2};
    epochs = animal_info{session_list,3};
    daystring = num2str(day);

    dir = 'W:\data\PPP\'; % base folder

    dir = [dir,animalprefix,'\day',daystring,'\']; % data folder
    prefix = ['day',daystring];
    disp([animalprefix,' Day-',num2str(day)])  % show progress
    warning('off')

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
    %% epoch loop, gather spike matrix
    for ep = 1:length(epochs)
        epoch = epochs(ep);
        %% get trial-by-trial firing rates
        trial_conditions = behavior.trialConds; % trajectory types/conditions
        % load the ratemaps
        load(fullfile(dir,  [prefix,'.firingMapsTrialsNorm.cellinfo.mat']));% load firing rate maps

        cell_count = 0; % reset
        dCA1_cellnum = length(spikes.UID);
        cellnum = length(firingMaps_trials);
        epochtimes = MergePoints.timestamps(epoch,:); % epoch start and end time

        % get trial info
        trialid_ep = find(behavior.trials(:,1) >= epochtimes(1) &  behavior.trials(:,2) <= epochtimes(2));
        trial_correct  = double(behavior.trialcorrect(trialid_ep));
        trialtime = mean(behavior.trials(trialid_ep,:),2);
        trial_condition = behavior.trialConds(trialid_ep);

        for i = 1:cellnum
            if ismember(firingMaps_trials{i}.UID, spikes.UID) % dorsal CA1 cells only
                cell_count = cell_count + 1;
                temp1 = firingMaps_trials{i}.rateMaps(trialid_ep(trial_condition == 1),:); % trajectory type 1
                temp2 = firingMaps_trials{i}.rateMaps(trialid_ep(trial_condition == 2),:); % trajectory type 2
                temp1(isnan(temp1)) = 0;
                temp2(isnan(temp2)) = 0;

                spike_matrix_trial{ep}(:,cell_count) = nanmean(temp1,2)'; % spike matrix for trial/time
                mintrial = min(length(temp1(:,1)),length(temp2(:,1))); % take the minimal trials
                spike_matrix_splitter{ep}(:,cell_count) = nanmean(temp1(1:mintrial,1:37) - temp2(1:mintrial,1:37),2)'; % spike matrix for choice/splitter (center stem only)
                spike_matrix_space{ep}(:,cell_count) = nanmean(temp1); % spike matrix for space/locations, EP1
                spike_matrix_space2{ep}(:,cell_count) = nanmean(temp2); % spike matrix for space/locations, EP2

                % shuffled spike matrices for location, circular shift of positions
                temp1_shuf = circshift(nanmean(temp1), randperm(100,1)); 
                spike_matrix_space_shuf{ep}(:,cell_count) = temp1_shuf;

                temp2_shuf = circshift(nanmean(temp2), randperm(100,1));
                spike_matrix_space2_shuf{ep}(:,cell_count) = temp2_shuf;
                
                % shuffled spike matrices for trial time, shuffle trial labels
                trialnum = length(temp1(:,1));
                spike_matrix_session_shuf{ep}(:,cell_count) = nanmean(temp1(randperm(trialnum,round(trialnum/2)),:));
                spike_matrix_session_shuf2{ep}(:,cell_count) = nanmean(temp1(randperm(trialnum,round(trialnum/2)),:));

                trialnum = length(temp2(:,1));
                spike_matrix_session2_shuf{ep}(:,cell_count) = nanmean(temp2(randperm(trialnum,round(trialnum/2)),:));
                spike_matrix_session2_shuf2{ep}(:,cell_count) = nanmean(temp2(randperm(trialnum,round(trialnum/2)),:));

            end
        end
    end
    %% get Principal Angles (PAs)
    % PA for EP1 location - EP2 location, trajectory type 1
    PA1 = getPrincipalAngle(spike_matrix_space{1}',spike_matrix_space{2}'); % real
    PA1_shuf =  getPrincipalAngle(spike_matrix_session_shuf{1}',spike_matrix_session_shuf2{1}'); %shuffled
    PA11_shuf = getPrincipalAngle(spike_matrix_space_shuf{1}',spike_matrix_space_shuf{2}');%shuffled

    % PA for EP1 location - EP2 location, trajectory type 2
    PA2 = getPrincipalAngle(spike_matrix_space2{1}',spike_matrix_space2{2}'); % real
    PA2_shuf =  getPrincipalAngle(spike_matrix_session2_shuf{1}',spike_matrix_session2_shuf2{1}'); %shuffled
    PA12_shuf = getPrincipalAngle(spike_matrix_space2_shuf{1}',spike_matrix_space2_shuf{2}'); %shuffled

    % PA for location -  choice
    PA12 = getPrincipalAngle(spike_matrix_space{1}',spike_matrix_splitter{1}');
    PA22 = getPrincipalAngle(spike_matrix_space{2}',spike_matrix_splitter{2}');

    % PA for location -  time
    PA23 = getPrincipalAngle(spike_matrix_space{2}',spike_matrix_trial{2}');
    PA13 = getPrincipalAngle(spike_matrix_space{1}',spike_matrix_trial{1}');


    % remove shuffled residual from the real angle
    if drift_remap
        PA1 =  abs(PA1 - PA1_shuf);
        PA2 =  abs(PA2 - PA2_shuf);
    else
        PA1 =  90 - abs(PA1 - PA11_shuf);
        PA2 =  90 - abs(PA2 - PA12_shuf);
    end

    PA12 = 90 - abs(PA12 - PA11_shuf);
    PA13 = 90 - abs(PA13 - PA11_shuf);

    PA22 = 90 - abs(PA22 - PA12_shuf);
    PA23 = 90 - abs(PA23 - PA12_shuf);

    % gather results
    PA_gather = [PA_gather;PA1,PA2,PA12,PA13,PA22,PA23];

    clear spike_matrix_trial; clear spike_matrix_splitter; clear spike_matrix_space; clear spike_matrix_space2
end


        

