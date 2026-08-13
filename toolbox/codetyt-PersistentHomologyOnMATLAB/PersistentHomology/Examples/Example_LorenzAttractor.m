close all; clc; clear;

% Example 
dat = load("/Users/wt248/Documents/MATLAB/codetyt-PersistentHomologyOnMATLAB/PersistentHomology/Dat/Lorenz-chaos-dt0.001-T1000.mat");
ndata = 1e2;
data = dat.x(1000:50:(1000+ndata*50), :);
[PD, Rinfs] = get_PD_H012_from_3Ddata(data);

% Visualization
figure;
scatter3(data(:, 1), data(:, 2), data(:, 3), 'filled'); hold on
VisualizePersistentDiagram(PD, Rinfs);
