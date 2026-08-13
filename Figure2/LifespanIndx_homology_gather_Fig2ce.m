clc
clear all
close all
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\codetyt-PersistentHomologyOnMATLAB')) %(co)homology toolbox
%% data list
% familiar-novel (M1-M2), day 1, [animal, day, epoch, dimensions]
animal_info = [{'PPP7'},{12},{3},{[1,2]};...
    {'PPP7'},{12},{5},{[1,2]};...
    {'PPP8'},{12},{2},{[2,3]};...
    {'PPP8'},{12},{4},{[1,3]};...
    {'PPP20'},{7},{2},{[1,3]};...
    {'PPP20'},{7},{4},{[1,3]};...
    {'PPP21'},{7},{2},{[1,3]};...
    {'PPP21'},{7},{4},{[2,3]};...
    {'PPP23'},{8},{2},{[1,2]};...
    {'PPP23'},{8},{4},{[1,2]}];

% novel maze 1 (M2), day 2,[animal, day, epoch, dimensions]
% animal_info = [{'PPP7'},{14},{4},{[1,2]};...
%     {'PPP8'},{14},{4},{[1,2]};...
%     {'PPP20'},{8},{4},{[1,3]};...
%     {'PPP21'},{8},{4},{[1,3]};...
%     {'PPP23'},{9},{4},{[2,3]};];

% % novel maze 2 (M3), day 1, [animal, day, epoch, dimensions]
% animal_info = [{'PPP8'},{15},{6},{[1,3]};...
%     {'PPP20'},{13},{6},{[1,3]};...
%     {'PPP21'},{13},{6},{[1,2]};...
%     {'PPP23'},{11},{6},{[1,3]}];
%% set parameters
nbins = 100; % downsample to 100 points for robust homology calculation
nRuns = 500; % number of times to resample data 
sample_prc = 0.8; % resample by taking 80% data
%% gather data
session_num = 0; %reset
for session_list = 1:length(animal_info)
    session_num = session_num +1;
    animalprefix = animal_info{session_list,1};
    day = animal_info{session_list,2};
    epoch = animal_info{session_list,3};
    dims = animal_info{session_list,4};

    dir = 'W:\data\PPP\'; % data folder

    daystring = num2str(day);
    animaldir = [dir,animalprefix,'\day',daystring,'\']; % data folder
    prefix = ['day',daystring];
    disp([animalprefix,' Day-',num2str(day)]) % show progress

    % load preprocessed files
    filename = [animalprefix,'D',daystring,'EP', num2str(epoch), '_topology_result.mat']; % file name
    load(filename)
    %% downsample for robust homology calculation, random sample with replacement
    % get the centroid for each position bin for robust estimation of topology
    nSamples = length(validid);
    maxpos = ceil(max(linpos_RUN(:,1)));
    minpos = floor(min(linpos_RUN(:,1)));
    posbin = minpos:(maxpos-minpos)/nbins:maxpos;
    [~,ind] = histc(linpos_RUN(validid,1),posbin);

    for run = 1:nRuns
        s = RandStream('dsfmt19937','Seed',run); %set a seed for reproducibility
        spike_umap_avg = nan(nbins, length(spike_umap(1,:)));
        for i = 1:nbins
            currentid = find(ind == i);
            Nind = length(currentid);
            resample_IDs = randperm(s,Nind);
            N_resample = 1:round(sample_prc*Nind);
            current_resampleIDs = currentid(resample_IDs(N_resample));
            temp = nanmedian(spike_umap(current_resampleIDs,:)); 
            spike_umap_avg(i,:) = temp;
        end
    
        % remove NaN (empty bins)
        temp = sum(spike_umap_avg,2);
        nonNaN = find(~isnan(temp));
        spike_umap_avg = spike_umap_avg(nonNaN,:);
        spike_matrix_label = posbin(nonNaN+1);

        % homology
        aa = find(spike_matrix_label > 0);
        [PD, Rinfs] = get_PD_H01_from_2Ddata(spike_umap_avg(aa,dims)); % calculate persistence diagram (PD), H0 and H1
        PD_length = PD{2}(:,2) - PD{2}(:,1); % bar length
        [PD_sorted,~] = sort(PD_length,'descend'); % sort bar lengths
        PD_ratio(run) = PD_sorted(1)/(PD_sorted(1)+PD_sorted(2)); % lifespan index, take the first and second largest loop
    end
    
    % gather results
    PD_ratio_gather{session_num} = PD_ratio';
    clear PD_ratio;
end















