clc
clear all
close all
%% add codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
%% list of data, [animal,session name, epochs]
% % uPSEM sessions
SesssionCondition = 'Final';
animal_info = [{'IZ29'},{'IZ29_1008um_201108_sess4'},{[1,4]};...
               {'IZ29'},{'IZ29_1008um_201110_sess6'},{[1,4]};...
               {'IZ29'},{'IZ29_1008um_201113_sess9'},{[1,4]};...
               {'IZ32'},{'IZ32_1008um_210303_sess7'},{[1,4]};...
               {'IZ33'},{'IZ33_580um_210312_sess8'},{[1,4]};...
               {'IZ33'},{'IZ33_580um_210315_sess9'},{[1,4]};...
               {'IZ33'},{'IZ33_580um_210317_sess10'},{[1,4]};...
               {'IZ33'},{'IZ33_580um_210319_sess11'},{[1,4]};...
               {'IZ34'},{'IZ34_436um_210311_sess7'},{[1,4]};...
               {'IZ34'},{'IZ34_436um_210315_sess8'},{[1,4]};...
               {'IZ34'},{'IZ34_436um_210317_sess9'},{[1,4]}];

% Saline sessions
% SesssionCondition = 'Saline';
% animal_info = [{'IZ29'},{'IZ29_1008um_201109_sess5'},{[1,4]};...
%                {'IZ29'},{'IZ29_1008um_201111_sess7'},{[1,4]};...
%                {'IZ32'},{'IZ32_1008um_210301_sess5'},{[1,4]};...
%                {'IZ32'},{'IZ32_1008um_210302_sess6'},{[1,4]};...
%                {'IZ33'},{'IZ33_580um_210304_sess3'},{[1,4]};...
%                {'IZ33'},{'IZ33_580um_210308_sess5'},{[1,4]};...
%                {'IZ33'},{'IZ33_580um_210309_sess6'},{[1,4]};...
%                {'IZ34'},{'IZ34_436um_210304_sess4'},{[1,4]};...
%                {'IZ34'},{'IZ34_436um_210305_sess5'},{[1,4]};...
%                {'IZ34'},{'IZ34_436um_210310_sess6'},{[1,4]}];
%% set parameters
pos_binnum = 100; % position bins
%% gather results
PVcorr_all = []; % gather, reset
for session_list = 1:length(animal_info)
    dir = 'U:\data\Zutshi_Neuron2021\CA3_mEC\'; % data folder
    animalname = animal_info{session_list,1};
    animalprefix = animal_info{session_list,2};
    epochs = animal_info{session_list,3};
    dir = [dir,animalname,'\',SesssionCondition,'\',animalprefix,'\'];
    disp([animalname,' ',animalprefix]) % show progress

    % load ratemaps
    load(fullfile(dir,  [animalprefix,'.firingMapsAvg_linear.cellinfo.mat']));% load firing rate maps
    %% calculate PVO (population vector overlap/correlation)
    % load CA1 PYR spikes
    spikes = importSpikes_legacy('basepath',dir,'CellType',"Pyramidal Cell",'brainRegion',"CA1");
    cellnum = length(spikes.UID);
    for e = 1:length(epochs)
        ep = epochs(e);
        rm1 = []; % ratemap matrix
        for i = 1:cellnum
            UID = spikes.UID(i);
            cind = find(firingMaps_linear{ep}.UID == UID);  
            if ~isempty(cind)
                linfield1 = firingMaps_linear{ep}.rateMaps{cind};  
            else
                linfield1=[];
            end

            if ~isempty(linfield1)
                linfield_cell = [];
                for track = 1:2
                    linfield_hp = linfield1{track};
                    a = find(linfield_hp < 0);
                    linfield_hp(a) = nan;
                    linfield_cell = [linfield_cell,linfield_hp];
                end
                if max(linfield_cell) < 2 % remove cells with a peak rate < 2Hz
                    linfield_cell = [nan(size(1:pos_binnum)),nan(size(1:pos_binnum))];
                end
            else
              linfield_cell = [nan(size(1:pos_binnum)),nan(size(1:pos_binnum))];
            end
            linfield_cell = linfield_cell - nanmean(linfield_cell);

            rm1 = [rm1;linfield_cell];
        end
        rm{e} = rm1;
    end 

    % real PVO
    for i = 1:length(rm{e}(1,:))
        corr12(i) = corr(rm{1}(:,i),rm{2}(:,i),'rows','complete');
    end

    % shuffled PVO
    for run = 1:100
        randind = randperm(length(rm{2}(:,1)));
        for i = 1:length(rm{e}(1,:))
            corr12_shuf(run,i) = corr(rm{1}(:,i),rm{2}(randind,i),'rows','complete');
        end
    end
    % 95th percentile of shuffled PVO
    corr12_shuf_95 = prctile(corr12_shuf,95);
    corr12 = corr12';
    corr12_shuf = corr12_shuf';
    PVcorr_all = [PVcorr_all;nanmean(corr12), nanmean(corr12_shuf_95)]; % gather, [real, 95th percentile of shuffle]
    clear corr12
    clear corr12_shuf
end
