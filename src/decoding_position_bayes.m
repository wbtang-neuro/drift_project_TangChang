function [accuracy,decodinginfo] = decoding_position_bayes(dir,ep,spikes,linfields,linfields_pos)
%decoding_position_bayes - Bayesian decoding of animals' position using
%prefined place-field template (i.e., linfields).
%
%  USAGE
%
%  EXAMPLE:
%  decoding_position_bayes('/Volumes/Drift_Proj/PPP_PVR/PPP7/day12/',5,spikes,linfields,linfields_pos) 
%
%    dir              data folder (i.e., basepath)
%    ep               testing epoch for decoding
%    spikes           spikes for decoding
%    linfields        linearized place-field template
%    linfields_pos    corresponding position info for the place-field template 
%
% OUTPUTS
%
%    accuracy         decoding accuracy
%    decodinginfo     struct, bin-by-bin decoding results
%    =========================================================================
% 2025 Wenbo Tang

tBinSz = 500; %default temporal bin in ms
tBinSz_sm = 1000; %default sliding temporal bin in ms

%----- neurons index-----%
hpnum = length(spikes.UID);
track_num = length(linfields{1}); %trajectory number
trackpos = linfields_pos';
%% select spatially modulated cells
%-----create the ratemaps [nPosBin x nHPCells]-----%
rm = []; % ratemap matrix
pm = []; % position matrix
tm = []; % track matrix
cellidxm = [];

for i = 1:hpnum
    linfield1 = linfields{i};
    linfield_hp = [];
    lintrack_hp = [];
    pos_hp = [];
    for track = 1:track_num % trajectory types, 2 = left and right
        occnormrate1 = linfield1{track}';
        lintrack1 = ones(size(trackpos))*track;
        linfield_hp = [linfield_hp;occnormrate1];
        pos_hp = [pos_hp;trackpos];
        lintrack_hp = [lintrack_hp;lintrack1];
    end
     
   if (max(linfield_hp) >= 3) % overall peak firing rate larger than 3 Hz
      a = find(linfield_hp < 0); % -1 for bin without enough occupancy
      %pad nan
      if ~isempty(a)
           [lo,hi]= findcontiguous(a);  %find contiguous NaNs
           for ii = 1:length(lo) 
               if lo(ii) > 1 & hi(ii) < length(linfield_hp)
                   fill = linspace(linfield_hp(lo(ii)-1), ...
                           linfield_hp(hi(ii)+1), hi(ii)-lo(ii)+1);
                   linfield_hp(lo(ii):hi(ii)) = fill;
               end
           end
      end
      rm = [rm;linfield_hp'];
      pm = [pm;pos_hp'];
      tm = [tm;lintrack_hp'];
      cellidxm = [cellidxm; i];
    end
end

rm = rm'; %[nPosBin x nHPCells],  rate map
pm = pm'; % position bins for the rate map 
tm = tm'; % trajectory type for the rate map
trajinfo = mean(tm,2);% trajactory type
rm = rm+ (eps.^8); %Add a small number so there are no zeros
expecSpk =rm.*tBinSz_sm./1000; %[nPos x nCells] Expected number of spikes per bin
hpnum = length(rm(1,:)); % update
%% get actual position infomation during running periods for the testing epoch
%load position info
load(fullfile(dir, [basenameFromBasepath(dir),'.posTrials_speedfiltered.cellinfo.mat']));% posTrials, running periods only
% trajectory type 1
linpos1 = posTrials_speedfiltered{ep}{1};
linpos1(:,2) = linpos1(:,2)  - min(linpos1(:,2));
linpos1(:,2) = linpos1(:,2) ./max(linpos1(:,2)); % use normalized position
linpos1 = [linpos1,ones(length(linpos1(:,1)),1)];%(time,linpos,traj)
% trajectory type 2
linpos2 = posTrials_speedfiltered{ep}{2};
linpos2(:,2)  = linpos2(:,2)  - min(linpos2(:,2));
linpos2(:,2)  = linpos2(:,2)./max(linpos2(:,2)); % use normalized position
linpos2 = [linpos2,2.*ones(length(linpos2(:,1)),1)];%(time,linpos,traj)
linpos = [linpos1;linpos2];

% sort by time
[~,indx] = sort(linpos(:,1));
linpos = linpos(indx,:);

lintimes = linpos(:,1); % time 
traj = linpos(:,3); % trajectory type
lindist = linpos(:,2); % linearized position
timevec = lintimes(1):tBinSz/1000:lintimes(end); % time vector
%% Bin-by-bin Bayesian decoding
predict_gather = []; % gather result
decodinginfo = struct; % reset
for i = 1:length(timevec)
    bin = timevec(i);
    avglindist_bin = find(lintimes >= (bin - tBinSz_sm/2000) & lintimes < (bin + tBinSz_sm/2000)); % current bin
    if ~isempty(avglindist_bin)
        traj_bin = traj(avglindist_bin);
        if all(traj_bin-traj_bin(1) == 0) && (traj_bin(1) > 0)%valid bin
           current_tr = traj_bin(1);
           avglindist = nanmean(lindist(avglindist_bin)); % average position
           
           %% gather spikes in that bin
           celldata = [];
           for n = 1:hpnum
              index = cellidxm(n) ;
              if ~isempty(spikes.times{index})
                  spiketimes = spikes.times{index};
              else
                  spiketimes = [];
              end
              if ~isempty(spiketimes)
                  spikebins = spiketimes(find(spiketimes >= (bin - tBinSz_sm/2000) & spiketimes <= (bin + tBinSz_sm/2000)));
                  spikecount(n) = length(spikebins);
                  tmpcelldata = [length(spikebins),n];
              else
                  tmpcelldata = [0,n];
                  spikecount(n) = 0;
              end
              celldata = [celldata;tmpcelldata];
           end

           cellcounts = sum((spikecount > 0));
           trajidx = find(trajinfo == current_tr); % restrict to the current trajectory
           pm_tr = pm(trajidx,1);
           
           if (cellcounts > 4) % minimal 4 cell active
              cellsi = find(spikecount > 0);
              %% Bayesian
              rm_tr = rm(trajidx,cellsi);
              spkPerBin = celldata(cellsi,1)';
              expecSpk_tr = expecSpk(trajidx,cellsi);
              nPBin = size(rm_tr,1); %N positional bin
              wrking = bsxfun(@power, rm_tr, spkPerBin); %[nPos x nTbin x nCell]
              wrking = prod(wrking,2);
              expon = exp(-sum(expecSpk_tr,2)); %Exponent of equation.
              post = bsxfun(@times,wrking, expon); %[nPos x nTbin x nCell]
              post(isnan(post)) = 0;  
              post = post./sum(post); %normalize
              [~,id] = max(post); % bin with maximal posterior prob
              errorpos = pm_tr(id) - avglindist; % prediction error
             
              % gather info
              decodinginfo(i).bin = bin;
              decodinginfo(i).traj = current_tr;              
              decodinginfo(i).actualpos = avglindist;
              decodinginfo(i).decodedpos = pm_tr(id);
              decodinginfo(i).error = errorpos;
              decodinginfo(i).pMat = post;
              predict_gather = [predict_gather;avglindist,pm_tr(id)]; % [actual position, predicted position]
           else
              decodinginfo(i).bin = bin;
              decodinginfo(i).traj = NaN;              
              decodinginfo(i).actualpos = NaN;
              decodinginfo(i).decodedpos = NaN;
              decodinginfo(i).error = NaN;
              decodinginfo(i).pMat = [];
           end
       else
          decodinginfo(i).bin = bin;
          decodinginfo(i).traj = NaN;              
          decodinginfo(i).actualpos = NaN;
          decodinginfo(i).decodedpos = NaN;
          decodinginfo(i).error = NaN;
          decodinginfo(i).pMat = [];
       end

    else
       decodinginfo(i).bin = bin;
       decodinginfo(i).traj = NaN;              
       decodinginfo(i).actualpos = NaN;
       decodinginfo(i).decodedpos = NaN;
       decodinginfo(i).error = NaN;
       decodinginfo(i).pMat = [];
    end
end
% prediction accuracy as correlation between actual and predict positions
accuracy = corr(predict_gather(:,1),predict_gather(:,2),'type','spearman','rows','complete');

       


       
          

              
            
            


        
        
        
        
