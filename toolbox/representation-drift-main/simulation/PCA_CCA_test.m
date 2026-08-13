clear
% close all
clc
rng(100)  % set the default random number generator seed
%% setting for the graphics
defaultGraphicsSetttings

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

%% Generate sample data, with small N
scale_factor = 1;

params.dim_out = 100;           % number of neurons
params.dim_in = 2;              % input dimensionality, 3 for Tmaze and 2 for ring
total_iter = 1e4;               % total simulation iterations
params.noiseStd = 0.001;         % 0.005 for ring, 1e-3 for Tmaze
k = params.dim_out;
n = params.dim_in;

% generate the ring data input, 2D
num_angle = 1e2;                % total number of angles
t = num_angle;
X = generate_ring_input(num_angle);
% X = X./0.5;
% X = [0:1/(num_angle-1):1;zeros(1,num_angle)];
% X = [zeros(1,num_angle);0:1/(num_angle-1):1];

% Cx = X*X'/num_angle;      % input sample covarian
% X(1,:) = X(1,:) + rand(1,1);
% X(2,:) = X(2,:) + rand(1,1);

% select one input to track, we compre the representations in 3 models
% xsel = X(:,randperm(t,1)); 
xsel = X;
%% setup the learning parameters
params.noiseStd = 0.001;         % 0.005 for ring, 1e-3 for Tmaze
learnRate = 0.02;        % default 0.05
dt = learnRate;

tot_iter = 200;     % total number of updates
step = 1;          % store every 10 updates
PSP_PCA = 1;
%%
if PSP_PCA 
    W_psp = 0.1*randn(params.dim_out,params.dim_in);
    M_psp = eye(params.dim_out); % lateral connection if using simple nsm
    noise1 = params.noiseStd* scale_factor;
%     noise1 = 0;
    noise2 = params.noiseStd;
%     noise2 = 0;


    dt0 = dt;          % learning rate for the initial phase, can be larger for faster convergence
    for i = 1:500
        Y = pinv(M_psp)*W_psp*X; 
        W_psp = (1-dt0)*W_psp + dt0*Y*X'/t + sqrt(dt)*noise1*randn(k,n);
        M_psp = (1-dt0)*M_psp + dt0*(Y*Y')/t + sqrt(dt)*noise2*randn(k,k);
        F_psp = pinv(M_psp)*W_psp;
        disp(norm(F_psp*F_psp'-eye(k),'fro'))
    end

    time_points = round(tot_iter/step);
    Yt_psp = [];
    pspErr = nan(time_points,1);
    % Ytest = zeros(k,time_points,num_test);  % store the testing stimulus

    for i = 1:tot_iter
        curr_inx = randperm(t,1);  % randm generate one sample
        % generate noise matrices
        Y = pinv(M_psp)*W_psp*X(:,curr_inx); 
        W_psp = (1-dt)*W_psp + dt*Y*X(:,curr_inx)' + sqrt(dt)*noise1*randn(k,n);
        M_psp = (1-dt)*M_psp + dt*Y*Y' +  sqrt(dt)*noise2*randn(k,k);

        if mod(i,step)==0
            temp = pinv(M_psp)*W_psp;       % current feature map
            Yt_psp = [Yt_psp,temp*xsel];        
        end  
    end
else
    %% Sanger's learning rule with only forward matrix

    Ws = randn(k,n);

    % Add a mask to M in each step
    temp = ones(k);
    % F = pinv(M)*W; % inital feature map
    noise1 = params.noiseStd * scale_factor;
    % noise2 = noiseStd;
    dt0 = dt;          % learning rate for the initial phase, can be larger for faster convergence
    for i = 1:500
        Y = Ws*X; 
        Ws = Ws + dt0*(Y*X'/t - tril(Y*Y'/t)*Ws);
    end
    %%

    time_points = round(tot_iter/step);
    % Yt_sg = zeros(k,time_points);
    Yt_sg = [];

    for i = 1:tot_iter
        curr_inx = randperm(t,1);  % randm generate one sample
        % generate noise matrices
        xis = randn(k,n);
        Y = Ws*X(:,curr_inx); 
        Ws =  Ws + dt*(Y*X(:,curr_inx)' - tril(Y*Y')*Ws) + sqrt(dt)*noise1*xis;

        if mod(i,step)==0
            Yt_sg = [Yt_sg,Ws*xsel];
        end
    end
end

%%
% apply PCA
if PSP_PCA
    [~, spike_pca, ~, ~, explained] = pca(Yt_psp');
%     spike_pca = run_umap(Yt_psp','min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3);
else
    [~, spike_pca, ~, ~, explained] = pca(Yt_sg');
%     spike_pca = run_umap(Yt_sg','min_dist',0.6,'n_neighbors',50,'metric','cosine','n_components',3);
end
traj2 = viridis(num_angle*tot_iter-5000);
traj_colormap = traj2;
figure,scatter3(spike_pca(5001:end,1),spike_pca(5001:end,2),spike_pca(5001:end,3),30*ones(size(num_angle*tot_iter-5000,1)),(1:num_angle*tot_iter-5000)','filled')
colormap(traj_colormap)
%%
% [~, spike_pca, ~, ~, explained] = pca([Y1,Y]');
% traj2 = viridis(num_angle*2);
% traj_colormap = traj2;
% figure,scatter3(spike_pca(:,1),spike_pca(:,2),spike_pca(:,3),30*ones(size(num_angle*2,1)),(1:num_angle*2)','filled')
% colormap(traj_colormap)
%%
% calculate rotation angle
% separate the manifold by epoch
spike_pca1 = spike_pca(50*num_angle+1:51*num_angle,1:3);
for i = 51:tot_iter-1
%     spike_pca1 = spike_pca((i-1)*num_angle+1:i*num_angle,1:3);
    spike_pca2 = spike_pca(i*num_angle+1:(i+1)*num_angle,1:3);

    % apply procrustes transformation
    [d,Z,transform] = procrustes(spike_pca1,spike_pca2,'reflection', false);

    % calculate rotation angle
    theta_abs(i-50+1) = acos((trace(transform.T)-1)/2)./pi *180; % convert to degree
end
