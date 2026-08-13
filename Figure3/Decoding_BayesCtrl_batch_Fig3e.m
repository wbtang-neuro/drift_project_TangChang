% this script is for cross-maze decoding using the Bayesian decoder (Fig. 3e)
clc
clear all
close all
%% add base codes to path
addpath(genpath('/Users/wt248/Documents/Remote_HMM_files/AYALab_Code/')) % AYA lab neurocode
%% data list, [animal, day, template epoch - testing epoch]
% M1-M2 session pairs
animal_info = [{'PPP7'},{12},{[3,5]};...
    {'PPP8'},{12},{[2,4]};...
    {'PPP20'},{7},{[2,4]};...
    {'PPP21'},{7},{[2,4]};...
    {'PPP23'},{8},{[2,4]}];

% % M1-M1' session pairs, for control
% animal_info = [{'PPP7'},{8},{[2,4]};...
%     {'PPP8'},{8},{[2,4]};...
%     {'PPP20'},{5},{[2,4]};...
%     {'PPP21'},{5},{[2,4]};...
%     {'PPP23'},{7},{[2,4]}];
%% set parameters
conditions = 2; % trajectory types/conditions, 2 = left and right
savedata = 0; % save results?
%% gather results
acc_all = []; % reset, gather decoding accuracy
for session_list = 1:length(animal_info)
    animalname = animal_info{session_list,1};
    % data folder
    if strcmp(animalname,'PPP7') || strcmp(animalname,'PPP8')  % CA1 only animals
        filedir = '/Volumes/Drift_Proj/PPP_PVR/';
    else
        filedir = '/Volumes/Drift_Proj/CA3CA1/';% CA1-CA3 dual recording animals
    end
    
    day = animal_info{session_list,2};
    eps = animal_info{session_list,3};
    
    daystring = num2str(day);
    dir = [filedir,animalname,'/day',daystring,'/'];
    animalprefix = ['day',daystring];
   
    disp([animalname,' Day-',num2str(day)]) % show progress
    %% load data
    % session-averaged linearlized rate maps
    load(fullfile(dir,[animalprefix,'.firingMapsAvg_linear.cellinfo.mat']));% load firing rate maps
    
    % load CA1 PYR spikes
    if  strcmp(animalname,'PPP7')
        spikes = importSpikes('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"dCA1");
    elseif strcmp(animalname,'PPP4') || (strcmp(animalname,'PPP8') && day == 8) || (strcmp(animalname,'PPP8') && day == 15)||...
            (strcmp(animalname,'PPP8') && day == 14)||(strcmp(animalname,'PPP8') && day == 13)
        spikes = importSpikes('basepath',dir,'CellType',"Pyramidal Cell");
    else
        spikes = importSpikes('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");
    end
    spikes_select = spikes;

    animalname = [animalname,'-',animalprefix];

    %% place-field template
    count = 0;
    cellnum = length(firingMaps_linear{eps(1)}.rateMaps);
    for i = 1:cellnum
        if ismember(firingMaps_linear{eps(1)}.UID(i), spikes_select.UID) % dorsal CA1 cells only
            count = count + 1;
            linfields{count} = firingMaps_linear{eps(1)}.rateMaps{i};
            linfields_pos = (1:length(firingMaps_linear{eps(1)}.params.x))/length(firingMaps_linear{eps(1)}.params.x);% use normalized positions 
        end
    end
    %% use the place-field template from epoch 1 to decode positions during epoch 2
    % get decoding info
    [acc,decodinginfo] = decoding_position_bayes(dir,eps(2),spikes,linfields,linfields_pos);
    decodinginfo_CA1 = decodinginfo;
    acc_all = [acc_all;acc]; % gather decoding accuracy
    clear decodinginfo
    %% save data?
    if savedata
        savedir = '/Users/wt248/Position_decoding_20240916/'; % folder for saving
        if exist(savedir) ~= 7
            mkdir(savedir)
        end
        save(sprintf('%s%sdecodinginfo_umap_control_DAY%02d.mat', savedir,animalname,day), 'decodinginfo_CA1');
    end
end

