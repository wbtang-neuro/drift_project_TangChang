function [timebase,Zcrosscov_sm,peakcov,peakcov_time] = Pairwise_peakcov_fun(spikes1,spikes2,timelist)
% calculate theta covariance for a pair of cells
%
%  USAGE
%
%    [timebase,Zcrosscov_sm,peakcov,peakcov_time] = Pairwise_peakcov_fun(spikes1,spikes2,timelist)
%
%    spikes1        spike train of cell 1
%    spikes2        spike train of cell 2
%    timelist       list of [start stop] for all theta/running periods
%
%  OUTPUT
%
%    timebase       time axis for the cross-correlogram
%    Zcrosscov_sm   z-scored cross-covariance
%    peakcov        peak cross-covariance
%    peakcov_time   cross-covariance peak time
%
%  NOTE
%
%  Before running this code, mex the spikexcorrc.cpp in the 'utilities' subfolder first 
%
%    Wenbo Tang, Mar 18, 2025

bin = 0.01; % time bin size, 10 ms
tmax = 0.5; % time range, +/- 500ms for corrln
sw1 = bin*3; % smoothing window, for smoothing corrln

tmaxcc = 0.2; % -200 to 200 ms extent to calculate theta peak covariance, see Siapas 2005

if ~isempty(timelist)
    timedur = sum(timelist(:,2)-timelist(:,1));

    % restrict spikes to running periods
    spikes1 = spikes1(find(isExcluded(spikes1,timelist)));
    spikes2 = spikes2(find(isExcluded(spikes2,timelist)));

    % calculate cross-correlogram only for cells with more than 100 spikes
    if (length(spikes1)> 100 && length(spikes2) > 100)
           spkstime1 = double(spikes1);
           spkstime2 = double(spikes2);

           % cross-correlation, first mex the spikexcorrc.cpp in the 'utilities' subfolder
           xc = spikexcorr(spkstime1, spkstime2, bin, tmax);
           p1 = xc.nspikes1/timedur; p2 = xc.nspikes2/timedur; % Firing rate in Hz
           exp_p = p1*p2; % per sec
           crosscov = (xc.c1vsc2 ./ (bin*timedur))-exp_p;

           % Convert to covariance, see Siapas 2005
           factor = sqrt((bin*timedur) / exp_p);
           Zcrosscov = crosscov .* (factor);
           rawcorr = xc.c1vsc2 ./ sqrt(xc.nspikes1 * xc.nspikes2);

           nstd=round(sw1/(xc.time(2) - xc.time(1))); % will be 3 std
           g1 = gaussian(nstd, nstd);
           timebase = xc.time;
           bins_run = find(abs(timebase) <= tmaxcc); % +/- Corrln window

           % smooth the curve
           Zcrosscov_sm = smoothvect(rawcorr, g1);

           currthetacorr_sm = nanmean(Zcrosscov_sm,1);
           [peakcov,peakcov_id] = nanmax(currthetacorr_sm(bins_run)); % Already smoothened
           peakcov_time = timebase(bins_run(peakcov_id));
    else
        timebase = nan;
        Zcrosscov_sm = nan;
        peakcov = nan;
        peakcov_time = nan;
    end
else
    timebase = nan;
    Zcrosscov_sm = nan;
    peakcov = nan;
    peakcov_time = nan;
end




