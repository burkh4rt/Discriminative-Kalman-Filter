p = fileparts(mfilename('fullpath'));
cd(p)
addpath(genpath(p));

dataset = struct([]);
iset =1;

for h = 1:5 % loop over Flint files
    load(['Flint_2012_e',num2str(h),'.mat'])
    nSubject = length(Subject);
    % for each subject
    for i = 1:nSubject
        nTrial = length(Subject(i).Trial);
        nNeuron = length(Subject(i).Trial(1).Neuron);
        vel = [];
        spk = [];
        % agglomerate over trials
        for j = 1:nTrial
            beats = length(Subject(i).Trial(j).Time);
            vel = [vel; Subject(i).Trial(j).HandVel];
            spk_k = zeros(beats,nNeuron);
            for k = 1:nNeuron
                spk_k(:,k) = histcounts(Subject(i).Trial(j).Neuron(k).Spike,beats);
            end
            spk = [spk; spk_k];
        end
        vel = vel(:,1:2);
        
        % each dataset is now in one place
        dataset(iset).velocities = vel;
        dataset(iset).spikes = spk;
        iset = iset + 1;
    end
    
    save('flint_gathered.mat','dataset');
    
end







