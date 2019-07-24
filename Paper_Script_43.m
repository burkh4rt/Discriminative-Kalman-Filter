p = fileparts(mfilename('fullpath'));
cd(p);
addpath(genpath(p));

% put the rng in the same state
rng(42,'twister'); tic

d = 10; % hidden state dimension
nList = round(10.^(0:.5:3)); % observation dimensions to explore

T = 10^4; % amount of testing data
mList = 10.^(1:5); % number of particles to explore for particle filter

%--- FAST VERSION ... COMMENT THIS TO REPRODUCE FIGURES IN TEXT ----
% T = 10^3;
% mList = 10.^(1:2);
%-------------------------------------------------------------------

% create parameters
S = eye(d);
A = .95*eye(d)-.05;
G = S-A*S*A.'; 

muZ = zeros(d,1);

R = 2; % mixture components
p = ones(R,1); p = p/sum(p);

nmax = max(10^3,max(nList));

Dbig = randi(d,nmax,1);
cbig = randn(nmax,1);
aconst = .01;
abig = repmat([aconst,1-aconst],nmax,1);
bbig = fliplr(abig);

% generate testing data
[z,xbig] = Paper_MakeData_53(T,S,A,G,d,p,cbig,Dbig,abig,bbig,muZ);

nN = numel(nList);
mN = numel(mList);

% output for predictions
zf = zeros(d,T,nN);
zDKF = zeros(d,T,nN,2);
zADF = zeros(d,T,nN);
zPF = zeros(d,T,nN,mN);
% output for times
tf = zeros(nN,1);
tDKF = zeros(nN,2);
tADF = zeros(nN,1);
tPF = zeros(nN,mN);

% loop over n
for k = 1:nN
    
    n = nList(k); fprintf(['\ncase n = ' num2str(n) ' ---------\n'])
    
    c = cbig(1:n);
    D = Dbig(1:n);
    a = abig(1:n,:);
    b = bbig(1:n,:);
    
    x = xbig(1:n,:);
    
    %-------------------------------------------
    % build the conditional means and variances
    %-------------------------------------------
    
    % true
    fprintf('f,Q: true ... '); tt = toc;
    
    [f,Q] = Paper_CondMeanVar_53(x,d,p,c,D,a,b,muZ,diag(S));

    zf(:,:,k) = f;
    
    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tf(k) = tt;
    
    %-------------------------------------------
    % predictions
    %-------------------------------------------
    
    % true DKF
    fprintf('DKF ... '); tt = toc;

    [zDKF(:,:,k,1),~,~,pdFlag] = Paper_DKF_Block(x,S,A,G,f-muZ,Q,false);
    zDKF(:,:,k,1) = zDKF(:,:,k,1)+muZ;
    
    if any(pdFlag > 0), fprintf([' pdFlag ',num2str(mean(pdFlag > 0)),'  ',num2str(mean(pdFlag(pdFlag > 0))),'  ']); end

    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tDKF(k,1) = tt;
    
    % robust DKF
    fprintf('robust DKF ... '); tt = toc;

    zDKF(:,:,k,2) = Paper_DKF_Block(x,S,A,G,f-muZ,Q,true)+muZ;

    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tDKF(k,2) = tt;
    
    
    % particle filters
    for j = 1:mN
        
        m = mList(j);
        
        fprintf(['particle filter (m = ' num2str(m) ') ... ']); tt = toc;
        
        zPF(:,:,k,j) = Paper_ParticleFilter_53(m,x,S,A,G,d,p,c,D,a,b,muZ);
        
        tt = toc-tt; fprintf('%0.1f s\n',tt);
        tPF(k,j) = tt;
        
    end % j loop over m
    
end % k loop over n


% compute RMSE
SS = sqrtm(S);
zS = SS\z;
r0 = sqrt(mean(mean(bsxfun(@minus,zS,0).^2,1),2));
rmu = sqrt(mean(mean(bsxfun(@minus,zS,mean(zS,2)).^2,1),2));

rf = zeros(nN,1);
rDKF = zeros(nN,2);
rPF = zeros(nN,mN);
for k = 1:nN
    rf(k) = sqrt(mean(mean((zS-SS\zf(:,:,k)).^2,1),2));
    rDKF(k,1) = sqrt(mean(mean((zS-SS\zDKF(:,:,k,1)).^2,1),2));
    rDKF(k,2) = sqrt(mean(mean((zS-SS\zDKF(:,:,k,2)).^2,1),2));
    for j = 1:mN
        rPF(k,j) = sqrt(mean(mean((zS-SS\zPF(:,:,k,j)).^2,1),2));
    end
end


% plotting
figure(1), clf
subplot(1,2,1)

ms = 8;
fs = 12;
pl1 = semilogx(nList,repmat(r0,size(rf)),'-','color','k','linewidth',2,'marker','.','markersize',ms); hold on
for k =1:size(rPF,2)
    pl2 = semilogx(nList,rPF(:,k),'-','color','b','linewidth',1,'marker','.','markersize',ms); hold on
    if exist('rPF2')==1
        semilogx(nList(1:numel(rPF2)),rPF2,'-','color','b','linewidth',1,'marker','.','markersize',ms); hold on
    end
end
pl4 = semilogx(nList,rf,':','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl5 = semilogx(nList,rDKF(:,2),'--','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl6 = semilogx(nList,rDKF(:,1),'-','color','r','linewidth',2,'marker','.','markersize',ms); hold on
set(gca,'fontsize',fs,'ylim',[0 1.25])
pl7 = legend([pl6,pl5,pl4,pl2,pl1],{'DKF','robust DKF','f(X_t)','particle filter','KF, EKF, UKF'});
ylabel('performance (RMSE)'), xlabel('observation dimensionality (n)')
hold off, figure(gcf)

subplot(1,2,2)
fsc = 1; % s per testing dataset
%fsc = 1000/T; % ms per timestep
for k =1:size(tPF,2)
    pl22 = loglog(nList,tPF(:,k)*fsc,'-','color','b','linewidth',1,'marker','.','markersize',ms); hold on
    if exist('tPF2')==1
        semilogx(nList(1:numel(tPF2)),tPF2,'-','color','b','linewidth',1,'marker','.','markersize',ms); hold on
    end
end
pl42 = loglog(nList,tf*fsc,':','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl52 = loglog(nList,(tf+tDKF(:,2))*fsc,'--','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl62 = loglog(nList,(tf+tDKF(:,1))*fsc,'-','color','r','linewidth',2,'marker','.','markersize',ms); hold on
set(gca,'fontsize',fs,'ylim',[10^-1.25 10^5])
pl72 = legend([pl62,pl52,pl42,pl22],{'DKF','robust DKF','f(X_t)','particle filter'});
ylabel('computation time (s)'), xlabel('observation dimensionality (n)')
hold off, figure(gcf)

% record results
save('43.mat');
