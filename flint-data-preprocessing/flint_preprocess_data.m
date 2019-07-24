p = fileparts(mfilename('fullpath'));
cd(p)
cd ../
addpath(genpath('dependencies'));
cd(p)

load('flint_gathered.mat');

nSets = length(dataset);
procd = struct([]);

for i = 1:nSets
    vel = dataset(i).velocities;
    spk = dataset(i).spikes;
    
    % downsample data to 100ms bins
    %vel_smooth = movmean(vel, [9, 0], 1);
    vel_ds = vel(5:10:end-5,:);
    
    spk_smooth = movsum(spk, [9, 0], 1);
    spk_ds = spk_smooth(10:10:end,:);
    
    % hit neural data with pca
    [~,spk_pca] = pca(spk_ds, 'NumComponents', 10);
    
    % zscore neural data
    spk_z = zscore(spk_pca, [], 1);
    
    procd(i).velocities = vel_ds;
    procd(i).spikes = spk_z;
end

save('flint_procd_alt.mat','procd');

