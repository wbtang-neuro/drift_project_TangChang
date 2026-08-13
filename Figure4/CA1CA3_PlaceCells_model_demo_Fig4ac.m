% This program tests how the receptive field change in a 1D ring place cell
% model (CA3-CA1).
% adapted from Qin et al. 2023
% Note: each run of the simulation will give you a slightly different
%       result, because of the randomness in synaptic noises.
close all
clear all
clc
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\representation-drift-main')) % base codes from Shanshan Qin 
%% setting for the graphics
defaultGraphicsSetttings

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');
%% ----PART 1: CA3 simulation---%
%% Generate sample data (get the CA3 place cells first)
params.dim_out = 50;           % number of CA3 neurons
params.dim_in = 3;              % input dimensionality, [x,y,context]
total_iter = 1e4;               % total simulation iterations

% default is a ring
dataType = 'ring';              
learnType = 'snsm';             % snsm if using simple non-negative similarity matching
noiseVar = 'same';              % using different noise or the same noise level for each synpase
params.batch_size = 1;          % default 1
params.record_step = 250;
save_data_flag = false;         % whether or not save the simulation data, default false

% generate the ring data input, 2D (2 rings)
radius = 1;
t = 2e3;           % total number of samples
X1 = generate_ring_input(t/2); % ring 1
X = [X1;ones(1,t/2)];  
X2 = [X1(2,:);X1(1,:);-ones(1,t/2)]; % ring 2
%% setup the learning parameters
params.noiseStd = 0.001;       % standard deviation of synaptic noise  
params.learnRate = 0.02;       % learning rate

Cx = X*X'/(t/2);  % input covariance matrix

% initialize the states
y0 = zeros(params.dim_out,params.batch_size);
Wout = zeros(1,params.dim_out); % linear decoder weight vector

params.W = 0.1*randn(params.dim_out,params.dim_in); 
params.M = 0.5*eye(params.dim_out); % lateral connection if using simple nsm;
params.lbd1 = 0.0;              % regularization for the simple nsm, 1e-3
params.lbd2 = 0.02;             % default 1e-3

params.alpha = 0;               % should be smaller than 1 if for the ring model
params.beta = 1;                % 
params.gy = 0.05;               % update step for y
params.b = zeros(params.dim_out,1);      % bias

params.sigWmax = params.noiseStd;    % the maxium noise level for the forward matrix
params.sigMmax = params.noiseStd*0.1;    % maximum noise level of recurrent matrix

% assume uniform distribution at log scale
if strcmp(noiseVar, 'various')
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
total_iter = 1000;    % number of conitnous iteration
params.record_step = 50;

[output1, params] = ring_update_weight_pf(X,total_iter,params); % first session on ring 1
[output2, ~] = ring_update_weight_pf(X2,total_iter,params); % second session on ring 2
output = output1;
step_1 = length(output1(1,1,:));
step_2 = length(output2(1,1,:));
output(:,:,step_1+1:step_1 + step_2) = output2;
all_Yts = output;
param_struct = params;
%% Analysis, check the change of CA3 place fields
time_points = size(all_Yts,3);
validid = 1:50;
Yt = all_Yts(validid,:,:);
% peak of receptive field
peakInx = nan(length(Yt(:,1,1)),time_points);
peakVal = nan(length(Yt(:,1,1)),time_points);
for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    peakInx(:,i) = peakPosi(:,1);
    peakVal(:,i) = pkVal(:,1);
end
% =========place field order by the begining ======================
% select three time points to compare the representations
inxSel = [1,10,11,20];
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
figure
% ======== ordered by the second input ==========
[~,neuroInx] = sort(peakInx(:,inxSel(3)));
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
%% ----PART 2: CA3-CA1 simulation---%
pFields = squeeze(all_Yts(:,:,10))';% ring 1
pFields = pFields./max(pFields)*0.9973;
pFields2 = squeeze(all_Yts(:,:,11))'; % ring2
pFields2 = pFields2./max(pFields2)*0.9973;
validid = find(max(pFields) > 0 & max(pFields2) > 0);

X2 = pFields2(:,validid)';

X = pFields(:,validid)';
dim_in = length(validid);
%% set up CA1 paramters
params.dim_out = 100;           % number of CA1 neurons
params.dim_in = dim_in;              % input dimensionality (input = CA3)
total_iter = 5e3;               % total simulation iterations

% default is a ring
dataType = 'ring';              
learnType = 'snsm';             % snsm if using simple non-negative similarity matching
noiseVar = 'same';              % using different noise or the same noise level for each synpase
params.batch_size = 1;          % default 1
params.record_step = 250;

params.noiseStd = 0.001;         % 0.005 for ring, 1e-3 for Tmaze
params.learnRate = 0.02;        % default 0.05

% initialize the states
y0 = zeros(params.dim_out,params.batch_size);
Wout = zeros(1,params.dim_out); % linear decoder weight vector
rand_seed = RandStream('mt19937ar','Seed',1);%seeds for reproducibility 

params.W = 0.5*randn(rand_seed,params.dim_out,params.dim_in);
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
[~, params] = ring_update_weight(X,total_iter,params);

% check the receptive field
Xsel = X(:,1:1:end);     % only use 10% of the data
Y0 = 0.1*rand(params.dim_out,size(Xsel,2));
Ys = nan(params.dim_out,size(Xsel,2));
for i = 1:size(Xsel,2)
    states_fixed_nn = MantHelper.nsmDynBatch(Xsel(:,i),Y0(:,i), params);
    Ys(:,i) = states_fixed_nn.Y;	
end
%% update after reaching steady state
total_iter = 1000;               % total simulation iterations
params.record_step = 50;
[output1, params1] = ring_update_weight(X,total_iter,params);
[output2, ~] = ring_update_weight(X2,total_iter,params1);
output = output1;
step_1 = length(output1(1,1,:));
step_2 = length(output2(1,1,:));
output(:,:,step_1+1:step_1 + step_2) = output2;
Yts = output;
%% plot CA1 place fields
time_points = size(Yts,3);
Yt = Yts;
% peak of receptive field
peakInx = nan(params.dim_out,time_points);
peakVal = nan(params.dim_out,time_points);
for i = 1:time_points
    [pkVal, peakPosi] = sort(Yt(:,:,i),2,'descend');
    peakInx(:,i) = peakPosi(:,1);
    peakVal(:,i) = pkVal(:,1);
end
% =========place field order by the begining ======================
% select three time points to compare the representations
inxSel = [1,10,11,20];
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
figure
% ======== ordered by the second input ==========
[~,neuroInx] = sort(peakInx(:,inxSel(3)));
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
%% -- PART3: CA3-CA1 assembly detection --%
load('ringPlaceModel_CA3CA1_S1.mat') % you can also run it anew (i.e., comment this out), this is just for demonstration purposes
%% get spike matrices
Yts1 = squeeze(Yts(:,:,10))'; % response to ring 1
Yts2 = squeeze(Yts(:,:,11))'; % response to ring 2
validid = find(max(Yts1) > 0.2 & max(Yts2) > 0.2); % find active cells
Yts1 = Yts1(:,validid);
Yts2 = Yts2(:,validid);

Yts1_CA3 = X';
Yts2_CA3 = X2';

CA1_num = length(Yts1(1,:));
CA3_num = length(Yts1_CA3(1,:));
%% SVD
[u0,s0,v0] = svd(params1.W(validid,:));
[eigenvalues,i] = sort(diag(s0),'descend');
CA1vec = u0(:,i);
CA3vec = v0(:,i);
%% significant eigenvalues
lambdaMax = 0.7/CA1_num; % Everitt and Dunn, 2001
relative_variances = eigenvalues.^2./(sum(eigenvalues.^2));
significant = relative_variances>lambdaMax;

CA1vec = CA1vec(:,significant);
CA3vec = CA3vec(:,significant);
assemble_num = sum(significant);
%% detect assembly memebers
for i = 1:assemble_num
    Template{i} = CA1vec(:,i)*CA3vec(:,i)';
    assemble_mean = nanmean(abs(CA1vec(:,i)));
    assemble_std = nanstd(abs(CA1vec(:,i)));
    threshold = assemble_mean + 2*assemble_std; % the weight matrix serves as the "ground truth" for the model, pick a low threshold. van de Ven 2016
    memeber_IDs = find(abs(CA1vec(:,i)) > threshold);
    assemble_members{i} = memeber_IDs;
end
%% calculate place-field COM shift for member pairs
COMshift_all = [];
COMshift_all_info = [];
posvec = (1:250)';
for group = 1:assemble_num
    memeber_IDs = assemble_members{group};
    memeber_num = length(memeber_IDs);
    if memeber_num > 1
        pairind = combnk(1:memeber_num,2);
    
        % calculate COM shift
        for pair = 1:length(pairind(:,1))
            cid1 = memeber_IDs(pairind(pair,1));
            cid2 = memeber_IDs(pairind(pair,2));
            CA1rate1 = Yts1(:,cid1);
            CA1rate2 = Yts2(:,cid1);
            COM1 = sum(CA1rate1.*posvec)/sum(CA1rate1);
            COM2 = sum(CA1rate2.*posvec)/sum(CA1rate2);
            [dCOM1,idx1] = min([COM1,250-COM1]);
            [dCOM2,idx2] = min([COM2,250-COM2]);
            d2 = min(COM1,250-COM1)+ min(COM2,250-COM2);
            d1 = abs(COM1 - COM2);
            [dist,type] = min([d1,d2]);
            if type == 1
                COMshift(1) = (COM1 - COM2);
            else
                if idx1 == 1
                    sign = 1;
                else
                    sign = -1;
                end
                COMshift(1) = d2 * sign;
            end

            CA1rate1 = Yts1(:,cid2);
            CA1rate2 = Yts2(:,cid2);
            COM1 = sum(CA1rate1.*posvec)/sum(CA1rate1);
            COM2 = sum(CA1rate2.*posvec)/sum(CA1rate2);
            [dCOM1,idx1] = min([COM1,250-COM1]);
            [dCOM2,idx2] = min([COM2,250-COM2]);
            d2 = min(COM1,250-COM1)+ min(COM2,250-COM2);
            d1 = abs(COM1 - COM2);
            [dist,type] = min([d1,d2]);
            if type == 1
                COMshift(2) = (COM1 - COM2);
            else
                if idx1 == 1
                    sign = 1;
                else
                    sign = -1;
                end
                COMshift(2) = d2 * sign;
            end
            COMshift_all = [COMshift_all;COMshift/250*2];
        end
    end
end
%% calculate place-field COM shift for all pairs
pairind = combnk(1:CA1_num,2);
COMshift_allpairs = [];
% calculate COM shift
for pair = 1:length(pairind(:,1))
    cid1 = pairind(pair,1);
    cid2 = pairind(pair,2);
    CA1rate1 = Yts1(:,cid1);
    CA1rate2 = Yts2(:,cid1);
    COM1 = sum(CA1rate1.*posvec)/sum(CA1rate1);
    COM2 = sum(CA1rate2.*posvec)/sum(CA1rate2);
    [dCOM1,idx1] = min([COM1,250-COM1]);
    [dCOM2,idx2] = min([COM2,250-COM2]);
    d2 = min(COM1,250-COM1)+ min(COM2,250-COM2);
    d1 = abs(COM1 - COM2);
    [dist,type] = min([d1,d2]);
    if type == 1
        COMshift(1) = (COM1 - COM2);
    else
        if idx1 == 1
            sign = 1;
        else
            sign = -1;
        end
        COMshift(1) = d2 * sign;
    end

    CA1rate1 = Yts1(:,cid2);
    CA1rate2 = Yts2(:,cid2);
    COM1 = sum(CA1rate1.*posvec)/sum(CA1rate1);
    COM2 = sum(CA1rate2.*posvec)/sum(CA1rate2);
    [dCOM1,idx1] = min([COM1,250-COM1]);
    [dCOM2,idx2] = min([COM2,250-COM2]);
    d2 = min(COM1,250-COM1)+ min(COM2,250-COM2);
    d1 = abs(COM1 - COM2);
    [dist,type] = min([d1,d2]);
    if type == 1
        COMshift(2) = (COM1 - COM2);
    else
        if idx1 == 1
            sign = 1;
        else
            sign = -1;
        end
        COMshift(2) = d2 * sign;
    end
    COMshift_allpairs = [COMshift_allpairs;COMshift/250*2];
end
%% correlation
rval = corr(COMshift_all(:,1),COMshift_all(:,2),'rows','complete');
rval_shuf = corr(COMshift_allpairs(:,1),COMshift_allpairs(:,2),'rows','complete');

corrfun = @(a,b)(corr(a,b,'rows','complete'));
bootstats1 = bootstrp(1000,corrfun,COMshift_all(:,1),COMshift_all(:,2));
bootstats2 = bootstrp(1000,corrfun,COMshift_allpairs(:,1),COMshift_allpairs(:,2));
%% plot results
figure
binedges = -1:0.1:1;
h = histogram2(COMshift_allpairs(:,1),COMshift_allpairs(:,2),binedges,binedges,'DisplayStyle','tile','ShowEmptyBins','on','Normalization','probability');
clim([0,0.005])
hold on
plot(COMshift_all(:,1),COMshift_all(:,2),'ro')
axis square
