function [Yt, Zt, params] = ring_update_weight_ExciInhi(X,total_iter,params,input_flag)
% flag indicating whether or not with input, default yes
if exist('input_flag','var')
    Flag = input_flag;
else
    Flag = true;
end

% update the synaptic weights and the stored population vectors if required
time_points = round(total_iter/params.record_step);
start_time = round(time_points/2);
Xsel = X(:,1:4:end);
Yt = nan(params.Np,size(Xsel,2),round(time_points/2)); % only use the second half
Zt = nan(params.Nin,size(Xsel,2),round(time_points/2)); % only use the second half
num_samp = size(X,2);

    for i = 1:total_iter
        if Flag
            inx = randperm(num_samp,params.batch_size);
            y0 = 0.1*rand(params.Np,params.batch_size);
            z0 = 0.1*rand(params.Nin,params.batch_size);
            x = X(:,inx);         % randomly select one input
            states = PlaceCellhelper.nsmDynBatchExciInhi(x,y0,z0, params);
            y = states.Y;
            z = states.Z;
        else
            x = zeros(params.Ng,1);
            y = zeros(params.Np,1);
            z = zeros(params.Nin,1);
        end
        % update weight matrix
        params.W = (1-params.learnRate)*params.W + params.learnRate*y*x'/params.batch_size + ...
            sqrt(params.learnRate)*params.noiseW*randn(params.Np,params.Ng);

        params.Wie =max((1-params.learnRate)*params.Wie + params.learnRate*z*y'/params.batch_size + ...
            sqrt(params.learnRate)*params.noiseWie*randn(params.Nin,params.Np),0);

        params.Wei = max((1-params.learnRate)*params.Wei + params.learnRate*y*z'/params.batch_size + ...
            sqrt(params.learnRate)*params.noiseWei*randn(params.Np,params.Nin),0);

        % notice the scaling factor for the recurrent matrix M
        params.M = max((1-params.learnRate)*params.M + params.learnRate*z*z'/params.batch_size + ...
            sqrt(params.learnRate)*params.noiseM*randn(params.Nin,params.Nin),0);

        params.b = (1-params.learnRate)*params.b + params.learnRate*sqrt(params.alpha)*mean(y,2);

        
        % store every param.step steps
        if mod(i, params.record_step) == 0 && i > round(total_iter/2)
            y0 = zeros(params.Np,size(Xsel,2));
            z0 = zeros(params.Nin,size(Xsel,2));
            states_fixed = PlaceCellhelper.nsmDynBatchExciInhi(Xsel,y0,z0, params);
            Yt(:,:,round(i/params.record_step)-start_time) = states_fixed.Y;
            Zt(:,:,round(i/params.record_step)-start_time) = states_fixed.Z;
        end
        
    end
end