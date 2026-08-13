close all; clc; clear;

% Example 
ndata = 1e2;
thetas = 2*pi*rand(ndata, 1);
data = [5*cos(thetas), 5*sin(thetas)];
data = data + 0.2*randn(size(data));
[PD, Rinfs] = get_PD_H01_from_2Ddata(data);

% Visualization
figure;
scatter(data(:, 1), data(:, 2), 'filled'); hold on
VisualizePersistentDiagram(PD, Rinfs);
