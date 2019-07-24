function zPF = SIR_PF(obs, weight_func, A, CZ, VZ, nparticles)
%SIR_PF implements sequential importance resampling particle filtering 
% method for approximating the Chapman-Kolmogorov equation via MC
% under the model:
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
        weights = weight_func(obs(:,t),particles);
        s = sum(weights);
        if s>0
            weights = weights/s;
        else
            warning(['bad weights at time ', num2str(t)]);
        end
        % average weighted particles
        zPF(:,t) = particles*weights;
        % resample particles
        idx = randsample(nparticles,nparticles,true,weights);
        particles = particles(:,idx);
    end
end
