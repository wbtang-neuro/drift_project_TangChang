% This program tests how the receptive field change in a 1D ring place cell
% model (CA1 only, with manipulation of lateral inhibition).
% adapted from Qin et al. 2023
% Note: each run of the simulation will give you a slightly different
%       result, because of the randomness in synaptic noises.
close all
clear all
clc
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\drift_project\toolbox\representation-drift-main\')) % base codes from Shanshan Qin 
%% setting for the graphics
defaultGraphicsSetttings

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');
%% Generate sample data, with small N
params.dim_out = 100;           % number of neurons
params.dim_in = 2;              % input dimensionality
total_iter = 1e4;               % total simulation iterations

% default is a ring
dataType = 'ring';              
learnType = 'snsm';             % snsm if using simple non-negative similarity matching
noiseVar = 'same';              % using different noise or the same noise level for each synpase
params.batch_size = 1;          % default 1
params.record_step = 250;

% generate the ring data input, 2D as in [x,y] position
num_angle = 1e3;                % total number of angles
X = generate_ring_input(num_angle);
%% setup the learning parameters
params.noiseStd = 0.001;         
params.learnRate = 0.02;        

Cx = X*X'/size(X,2);            % input covariance matrix

% initialize the states
y0 = zeros(params.dim_out,params.batch_size);
Wout = zeros(1,params.dim_out); % linear decoder weight vector
rand_seed = RandStream('mt19937ar','Seed',1);%seeds for reproducibility 

params.W = 0.1*randn(rand_seed,params.dim_out,params.dim_in);
params.M = eye(params.dim_out); % lateral connection if using simple nsm
params.lbd1 = 0.0;              % regularization for the simple nsm, 1e-3
params.lbd2 = 0.02;             % default 1e-3

params.alpha = 0;               % should be smaller than 1 if for the ring model
params.beta = 1;                % 
params.gy = 0.05;               % update step for y
params.b = zeros(params.dim_out,1);      % bias

params.sigWmax = params.noiseStd;    % the maxium noise level for the forward matrix
params.sigMmax = params.noiseStd;    % maximum noise level of recurrent matrix

% assume uniform distribution at log scale
if strcmp(noiseVar, 'various')
%     noiseVecW = rand(k,1);
    noiseVecW = 10.^(rand(params.dim_out,1)*2-2);
    params.noiseW = noiseVecW*ones(1,params.dim_in)*params.sigWmax;   % noise amplitude is the same for each posterior 
    noiseVecM = 10.^(rand(params.dim_out,1)*2-2);
    params.noiseM = noiseVecM*ones(1,params.dim_out)*params.sigMmax; 
else
    params.noiseW =  params.sigWmax*ones(params.dim_out,params.dim_in);   % stanard deivation of noise, same for all
    params.noiseM =  params.sigMmax*ones(params.dim_out,params.dim_out);   
end

% initial stage, make sure the weight matrices reach stationary state
[~, params] = ring_update_weight_pf(X,total_iter,params);

% check the receptive field
Xsel = X(:,1:1:end);     % only use 10% of the data
Y0 = 0.1*rand(params.dim_out,size(Xsel,2));
Ys = nan(params.dim_out,size(Xsel,2));
for i = 1:size(Xsel,2)
    states_fixed_nn = MantHelper.nsmDynBatch(Xsel(:,i),Y0(:,i), params);
    Ys(:,i) = states_fixed_nn.Y;	
end
%% continue updating with synaptic noise
for simulation_type = 1:2
    switch simulation_type
        % full noise model
        case simulation_type
            disp(simulation_type)
            if simulation_type == 1
                [output, ~] = ring_update_weight_Inh(X,total_iter,params,0); % control condition
            else
                [output, ~] = ring_update_weight_Inh(X,total_iter,params,1); % manipulate lateral inhibitory synapses
            end
            all_Yts{simulation_type} = output;
    end
end
%% load a previous simulation as an example?
load('ringPlaceModel_recordstep250_iter10000_S1.mat') % you can also run it anew (i.e., comment this out), this is just for demonstration purposes
%% Visualization of the neural manifold
sample_num = num_angle/4;
for simulation_type = 1:2
    Yt_psp = [];
    for i = 1:length(all_Yts{simulation_type}(1,1,:))
        temp = squeeze(all_Yts{simulation_type}(:,:,i));
        Yt_psp = [Yt_psp,temp];
    end
    [~, spike_pca, ~, ~, explained] = pca(Yt_psp');
    %%
    if simulation_type == 1
        traj2 = viridis(length(Yt_psp(1,:))); % control
    else
        traj2 = magma(length(Yt_psp(1,:))); % enhanced lateral inhition
    end

    traj_colormap = traj2;
    figure,scatter3(spike_pca(:,1),spike_pca(:,2),spike_pca(:,3),20*ones(length(Yt_psp(1,:)),1),(1:length(Yt_psp(1,:)))','filled')
    colormap(traj_colormap)
    title(['simulation type = ', num2str(simulation_type)])
    view([160,40])
    % calculate rotation angle
    spike_pca1 = spike_pca(1:sample_num,1:3);
    theta_abs = zeros(length(all_Yts{simulation_type}(1,1,:))-1,1);
    for i = 2:length(all_Yts{simulation_type}(1,1,:))-1
        spike_pca2 = spike_pca(i*sample_num+1:(i+1)*sample_num,1:3);

        % apply procrustes transformation
        [d,Z,transform] = procrustes(spike_pca1,spike_pca2,'reflection', false);

        % calculate rotation angle
        theta_abs(i) = acos((trace(transform.T)-1)/2)./pi *180; % convert to degree
    end
    theta_abs_all{simulation_type} = theta_abs;
end
%% Analysis, check the change of place field
pkThreshold = 0.05;  % active threshold
time_points = size(all_Yts{1},3);

for simulation_type = 1:2
    Yt = all_Yts{simulation_type};
    % peak of receptive field
    peakInx = nan(params.dim_out,time_points);
    peakVal = nan(params.dim_out,time_points);
    for i = 1:time_points
        [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
        peakInx(:,i) = peakPosi(:,1);
        peakVal(:,i) = pkVal(:,1);
    end

    % ======== faction of neurons have receptive field at a give time =====
    % quantified by the peak value larger than a threshold
    rfIndex = peakVal > pkThreshold;
    tolActiTime = mean(rfIndex,2);

    % fraction of neurons
    activeRatio = sum(rfIndex,1)/params.dim_out;
    
    % =========place field order by the begining ======================
    % select three time points to compare the representations
    inxSel = [1, 10, 11,20];
    figure
    [~,neuroInx] = sort(peakInx(:,inxSel(1)));
    for i = 1:length(inxSel)
        subplot(1,4,i)
        imagesc(Yt(neuroInx,:,inxSel(i)))
        colorbar
        title(['iteration ', num2str(inxSel(i))])
        ax = gca;
        ax.XTick = [1 500 1000];
        ax.XTickLabel = {'0', '\pi', '2\pi'};
        ylabel('neuron index')
        xlabel('position')
    end

    % ======== ordered by current index ==========
    figure
    for i = 1:length(inxSel)
        [~,neuroInx] = sort(peakInx(:,inxSel(i)));
        subplot(1,4,i)
        imagesc(Yt(neuroInx,:,inxSel(i)))
        colorbar
        ax = gca;
        ax.XTick = [1 500 1000];
        ax.XTickLabel = {'0', '\pi', '2\pi'};
        title(['iteration ', num2str(inxSel(i))])
        xlabel('position')
        ylabel('sorted index')
    end
end
%% plot PVO (population vector overlap/correlation) matrices
figure
m=100;
cm_viridis= viridis(m);
pkThreshold = 0.2;  % active threshold
for simulation_type = [1,2]
    Yt = all_Yts{simulation_type};
    Yt_avg = nanmean(Yt,3);
    [peakVal, peaklocs]  = max(Yt_avg');
    rfIndex = peakVal > pkThreshold;
    Yt = Yt(rfIndex,:,:);
    tr_number = length(Yt(1,1,:));
    pairind = combnk(1:tr_number,2);
    PVO_temp = ones(tr_number,tr_number);
    for pair = 1:length(pairind(:,1))
        trialpair = pairind(pair,:);
        trialno = trialpair;
        rm1 = squeeze(Yt(:,:,trialno(1)));
        rm2 = squeeze(Yt(:,:,trialno(2)));
        PVO_temp(trialpair(1),trialpair(2)) = corr(rm1(:),rm2(:),'rows','complete','type','Pearson');
        PVO_temp(trialpair(2),trialpair(1))  = corr(rm1(:),rm2(:),'rows','complete','type','Pearson');
    end
    subplot(1,2,simulation_type)
    imagesc(PVO_temp)
    caxis([0.1 0.9])
    axis square

    temp_tril = tril(PVO_temp,-1);
    validid = find(temp_tril);
    % gather PVO
    all_corr{simulation_type} = temp_tril(validid);
end




