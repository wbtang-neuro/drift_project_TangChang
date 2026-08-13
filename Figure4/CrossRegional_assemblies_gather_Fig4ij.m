close all
clear all;
clc
%% add base codes to path
addpath(genpath('C:\Users\Cornell\Documents\GitHub\neurocode')) % AYA lab neurocode
addpath(genpath('C:\Users\Cornell\Documents\Drift_Cell_Tang2025\toolbox\matplotlib')) % colormap toolbox
defaultGraphicsSetttings % graphic settings
%% define data
dir = 'W:\data\PPP\'; % data folder
% data list, [animal, day, M1 and M2 epochs]
animal_info = [{'PPP20'},{7},{[2,4]};...
    {'PPP20'},{8},{[2,4]};...
    {'PPP20'},{9},{[2,4]};...
    {'PPP20'},{10},{[2,4]};...
    {'PPP20'},{13},{[2,4]};...
    {'PPP20'},{14},{[2,4]};...
    {'PPP20'},{15},{[2,4]};...
    {'PPP21'},{7},{[2,4]};...
    {'PPP21'},{8},{[2,4]};...
    {'PPP21'},{9},{[2,4]};...
    {'PPP21'},{13},{[2,4]};...
    {'PPP21'},{14},{[2,4]};...
    {'PPP21'},{15},{[2,4]};...
    {'PPP23'},{8},{[2,4]};...
    {'PPP23'},{9},{[2,4]};...
    {'PPP23'},{10},{[2,4]}];
%% gather results
COMshift_members = []; % for gathering, reset
COMshift_allpair = [];
for session_list = 1:length(animal_info)
    animalname = animal_info{session_list,1};
    day = animal_info{session_list,2};
    eps = animal_info{session_list,3};
    daystring = num2str(day);
    prefix = ['day',daystring];
    animaldir = [dir,animalname,'\day',daystring,'\'];

    load(fullfile(animaldir, [basenameFromBasepath(animaldir),'.COMshift_allpairs.mat']));

    COMshift_members = [COMshift_members;(COMshift_all)];
    COMshift_allpair = [COMshift_allpair;(COMshift_allpairs)];
end
%% generate 2D histogram and scatter plots
% most data points are within the +/- 0.5 range, so take +/- 0.6 bin edges
validid = (abs(COMshift_allpair(:,1)) < 0.65 & abs(COMshift_allpair(:,2)) < 0.65);
binedges = -0.6:0.05:0.6;
figure
% raw histogram
h = histogram2(COMshift_allpair(validid,1),COMshift_allpair(validid,2),binedges,binedges,'DisplayStyle','tile','ShowEmptyBins','on','Normalization','probability');
COMshift_allpair_valid =COMshift_allpair(validid,:);
hist_prob = h.Values;
% smoothed histogram
hist_prob_re = imresize(hist_prob,[77,77])';
hist_sm = hist_prob_re;
hist_sm(find(hist_sm <=0)) = 0; % remove smoothing artifacts

% plot histgram
figure
set(gca,'YDir','normal')
xlim([-0.6,0.6])
ylim([-0.6,0.6])
axis square
hold on
plot(COMshift_allpair_valid(:,1),COMshift_allpair_valid(:,2),'.','color',[0.5,0.5,0.5])
% plot in log probability
binedges_re = -0.6:1.2/76:0.6;
imagesc(binedges_re,binedges_re,log10(hist_sm),'AlphaData',0.7)
contour(binedges_re,binedges_re,log10(hist_sm),10,'LineWidth',2) % add contours
colormap(viridis(100))
validid = (abs(COMshift_members(:,1)) < 0.65 & abs(COMshift_members(:,2)) < 0.65);
plot(COMshift_members(validid,1),COMshift_members(validid,2),'o','Color',[200,0,17]/255,'MarkerFaceColor',[243,169,164]/255)
%% bootstrapped correlation measure
SIcorrfun = @(a,b)(corr(a,b,'rows','complete'));
bootstats = bootstrp(1000,SIcorrfun,COMshift_members(validid,1),COMshift_members(validid,2)); % member pairs
bootstats_allpairs = bootstrp(1000,SIcorrfun,COMshift_allpair_valid(:,1),COMshift_allpair_valid(:,2)); % all pairs