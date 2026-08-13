clear
% close all
clc
rng(100)  % set the default random number generator seed
%% setting for the graphics
defaultGraphicsSetttings

rb = brewermap(11,'RdBu');
set1 = brewermap(9,'Set1');

%% Generate sample data, with small N
scale_factor = 0.5;
PSP_PCA = 0;

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

% Cx = X*X'/num_angle;      % input sample covarian
X2(1,:) = X(1,:) + rand(1,1);
X2(2,:) = X(2,:) + rand(1,1);
% X2(1,:) = X(1,:)*scale_factor;
% X2(2,:) = X(2,:)*scale_factor;
%% setup the learning parameters
params.noiseStd = 0.001;         % 0.005 for ring, 1e-3 for Tmaze
learnRate = 0.02;        % default 0.05
dt = learnRate;

%%
if PSP_PCA 
    W_psp = 0.1*randn(params.dim_out,params.dim_in);
    M_psp = eye(params.dim_out); % lateral connection if using simple nsm
    noise1 = params.noiseStd;
    noise2 = params.noiseStd;

    dt0 = dt;          % learning rate for the initial phase, can be larger for faster convergence
    for i = 1:500
        Y = pinv(M_psp)*W_psp*X; 
        W_psp = (1-dt0)*W_psp + dt0*Y*X'/t + sqrt(dt)*noise1*randn(k,n);
        M_psp = (1-dt0)*M_psp + dt0*(Y*Y')/t + sqrt(dt)*noise2*randn(k,k);
        F_psp = pinv(M_psp)*W_psp;
        disp(norm(F_psp*F_psp'-eye(k),'fro'))
    end
    
    W_psp = 0.1*randn(params.dim_out,params.dim_in);
    M_psp = eye(params.dim_out); % lateral connection if using simple nsm
    noise1 = params.noiseStd;
    noise2 = params.noiseStd;
    
    for i = 1:500
        Y2 = pinv(M_psp)*W_psp*X2; 
        W_psp = (1-dt0)*W_psp + dt0*Y2*X2'/t + sqrt(dt)*noise1*randn(k,n);
        M_psp = (1-dt0)*M_psp + dt0*(Y2*Y2')/t + sqrt(dt)*noise2*randn(k,k);
        F_psp = pinv(M_psp)*W_psp;
        disp(norm(F_psp*F_psp'-eye(k),'fro'))
    end
   
else
    %% Sanger's learning rule with only forward matrix

    Ws = randn(k,n);

    dt0 = dt;          % learning rate for the initial phase, can be larger for faster convergence
    for i = 1:500
        Y = Ws*X; 
        Ws = Ws + dt0*(Y*X'/t - tril(Y*Y'/t)*Ws);
    end
    %%
    Ws = randn(k,n);

    dt0 = dt;          % learning rate for the initial phase, can be larger for faster convergence
    for i = 1:500
        Y2 = Ws*X2; 
        Ws = Ws + dt0*(Y2*X2'/t - tril(Y2*Y2'/t)*Ws);
    end
end

%%
% apply PCA
[~, spike_pca, ~, ~, explained] = pca([Y,Y2]');
traj2 = viridis(num_angle*2);
traj_colormap = traj2;
figure,scatter3(spike_pca(:,1),spike_pca(:,2),spike_pca(:,3),30*ones(size(num_angle*2,1)),(1:num_angle*2)','filled')
colormap(traj_colormap)
%%
% % calculate rotation angle
% % separate the manifold by epoch
% spike_pca1 = spike_pca(50*num_angle+1:51*num_angle,1:3);
% for i = 51:tot_iter-1
% %     spike_pca1 = spike_pca((i-1)*num_angle+1:i*num_angle,1:3);
%     spike_pca2 = spike_pca(i*num_angle+1:(i+1)*num_angle,1:3);
% 
%     % apply procrustes transformation
%     [d,Z,transform] = procrustes(spike_pca1,spike_pca2,'reflection', false);
% 
%     % calculate rotation angle
%     theta_abs(i-50+1) = acos((trace(transform.T)-1)/2)./pi *180; % convert to degree
% end
