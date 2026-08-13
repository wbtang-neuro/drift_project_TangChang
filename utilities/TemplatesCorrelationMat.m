function [correlations,n] = TemplatesCorrelationMat(spikes,varargin)

% Part of the ActivityTemplates.m, only calculate the correlation matrix of
% neuronal population
%
%  USAGE
%
%    correlations = TemplatesCorrelationMat(spikes,<options>)
%
%    spikes         spike train (either single unit or MUA)
%    <options>      optional list of property-value pairs (see table below)
%
%    =========================================================================
%     Properties    Values
%    -------------------------------------------------------------------------
%     'bins'        list of [start stop] for all bins
%     'binSize'     bin size in s (default = 0.050)
%     'step'		step size in s (default = same as binSize)
%    =========================================================================
%
%  OUTPUT
%
%    correlations   a neuron-by-neuron correlation matrix over the provided
%                   bins, serving as basis for the analysis
%    n              zscored spike matrix over the provided bins
%
%  SEE
%
%    See also ActivityTemplates.m
%    Wenbo Tang, Dec 6, 2023
% Copyright (C) 2016-2022 by Michaël Zugaro, Ralitsa Todorova
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.

% Defaults
bins = [];
defaultBinSize = 0.050;
binSize = [];
step = [];

% Check number of parameters
if nargin < 1,
	error('Incorrect number of parameters (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).');
end
% Check parameter sizes
if ~isdmatrix(spikes,'@2'),
	error('Parameter ''spikes'' is not a Nx2 matrix (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).');
end

% Parse parameter list
for i = 1:2:length(varargin),
	if ~ischar(varargin{i}),
		error(['Parameter ' num2str(i+2) ' is not a property (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).']);
	end
	switch(lower(varargin{i})),
		case 'binsize',
			binSize = varargin{i+1};
			if ~isdscalar(binSize,'>0'),
				error('Incorrect value for property ''binSize'' (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).');
			end
		case 'step',
			step = varargin{i+1};
			if ~isdscalar(step,'>0'),
				error('Incorrect value for property ''step'' (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).');
			end
		case 'bins',
			bins = varargin{i+1};
			if ~isdmatrix(bins,'@2'),
				error('Incorrect value for property ''bins'' (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).');
			end
		otherwise,
			error(['Unknown property ''' num2str(varargin{i}) ''' (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).']);
	end
end

% Options binSize and bins are incompatible
if ~isempty(binSize) && ~isempty(bins),
	error('Parameters ''binSize'' and ''bins'' are incompatible (type ''help <a href="matlab:help ActivityTemplates">ActivityTemplates</a>'' for details).');
end
if isempty(binSize) && isempty(bins),
	binSize = defaultBinSize;
end
if isempty(step), step = binSize; end
nUnits = max(spikes(:,2));
correlations = nan(nUnits,nUnits);
if isempty(nUnits), return; end

%% Bin spikes
spikes = sortrows(spikes,1);
id = spikes(:,2);

% Shift spike times to start at 0, and list bins unless explicitly provided
if isempty(bins),
	spikes(:,1) = spikes(:,1) - spikes(1,1);
	bins = (0:step:(spikes(end,1)-binSize))';
	bins(:,2) = bins+binSize;
else
	m = min([min(spikes(:,1)) min(bins(:))]);
	spikes(:,1) = spikes(:,1) - m;
	bins = bins - m;
end

% Create spike count matrix
nBins = size(bins,1);
if isempty(nBins), return; end
n = zeros(nBins,nUnits);
for unit = 1:nUnits,
	n(:,unit) = CountInIntervals(spikes(id==unit,1),bins);
end

%% Create correlation matrix
n = zscore(n);
correlations = (1/(nBins-1))*n'*n;
