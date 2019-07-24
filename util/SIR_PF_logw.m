function zPF = SIR_PF_logw(obs, log_weight_func, A, CZ, VZ, nparticles)
%SIR_PF_LOGW implements sequential importance resampling particle filtering
% method in *log weight space*
% hidden(t)|hidden(t-1) ~ N(A*hidden(t-1), CZ)
% hidden(t) ~ N(0, VZ)
% weight_func(o,h) = p(observed(t)=o|hidden(t)=h), assumed vectorized
% uses nparticles

dz = size(A,1);
T = size(obs,2);
particles = mvnrnd(zeros([dz,1]),VZ,nparticles)';
zPF = zeros([dz,T]); % store estimates here

    for t = 1:T
        % update particle locations
        particles = A*particles + mvnrnd(zeros([dz,1]),CZ,nparticles)';
        % update weights
        logw = log_weight_func(obs(:,t),particles);
        weights = from_logs(logw);
        % average weighted particles
        zPF(:,t) = particles*weights;
        % resample particles
        idx = randsample(nparticles,nparticles,true,weights);
        particles = particles(:,idx);
    end
end

function wt = from_logs(logw)
    wt = max(logw);
    wt = exp(logw-wt);
    wt = wt/sum(wt);
end
