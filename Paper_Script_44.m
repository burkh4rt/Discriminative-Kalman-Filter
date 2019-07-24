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
Csig = [1,5]; 
c = numel(Csig); % number of mixture components
p = ones(c,1); p = p/sum(p);

nmax = max(10^3,max(nList));

Bbig = zeros(nmax,d,c); Bbig(:,:,1) = randn(nmax,d)/sqrt(d); Bbig(:,:,2) = -Bbig(:,:,1);
Cbig = zeros(nmax,nmax,c); Cbig(:,:,1) = Csig(1)*eye(nmax); Cbig(:,:,2) = Csig(2)*eye(nmax);

% generate testing data
hparam = 1;
[z,xbig] = Paper_MakeData_54(T,S,A,G,Bbig,Cbig,p,hparam);


nN = numel(nList);
mN = numel(mList);

% output for predictions
zf = zeros(d,T,nN);
zADF = zeros(d,T,nN);
zDKF = zeros(d,T,nN,2);
zPF = zeros(d,T,nN,mN);
% output for times
tf = zeros(nN,1);
tADF = zeros(nN,1);
tDKF = zeros(nN,2);
tPF = zeros(nN,mN);

% loop over n
for k = 1:nN
    
    n = nList(k); fprintf(['\ncase n = ' num2str(n) ' ---------\n'])
    
    B = Bbig(1:n,:,:);
    C = Cbig(1:n,1:n,:);
    x = xbig(1:n,:);
    
    %-------------------------------------------
    % build the conditional means and variances
    %-------------------------------------------
    
    % true
    fprintf('f,Q: true ... '); tt = toc;
    
    [f,Q] = Paper_CondMeanVar_52(x,S,B,C,p);
    
%     % test alternate code
%     [ff,QQ] = GaussMixMarg(x,0,S,p,0,B,C);
%     sum(abs(f(:)-ff(:)))
%     sum(abs(Q(:)-QQ(:)))
    
    zf(:,:,k) = f;
    
    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tf(k) = tt;
    
    %-------------------------------------------
    % predictions
    %-------------------------------------------
    
    % true DKF
    fprintf('DKF ... '); tt = toc;

    [zDKF(:,:,k,1),~,~,pdFlag,normFlag] = Paper_DKF_Block_54(x,S,A,G,f,Q,hparam,false);

    if any(pdFlag > 0), fprintf([' pdFlag ',num2str(mean(pdFlag > 0)),'  ',num2str(mean(pdFlag(pdFlag > 0))),'  ']); end
    if any(normFlag < 1), fprintf([' normFlag ',num2str(mean(normFlag < 1)),'  ',num2str(mean(normFlag(normFlag < 1))),'  ']); end
    
    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tDKF(k,1) = tt;
    
    % robust DKF
    fprintf('robust DKF ... '); tt = toc;

    [zDKF(:,:,k,2),~,~,pdFlag,normFlag] = Paper_DKF_Block_54(x,S,A,G,f,Q,hparam,true);

    if any(pdFlag > 0), fprintf([' pdFlag ',num2str(mean(pdFlag > 0)),'  ',num2str(mean(pdFlag(pdFlag > 0))),'  ']); end
    if any(normFlag < 1), fprintf([' normFlag ',num2str(mean(normFlag < 1)),'  ',num2str(mean(normFlag(normFlag < 1))),'  ']); end

    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tDKF(k,2) = tt;
    
    % G-ADF
    fprintf('G-ADF ... '); tt = toc;
    
    zADF(:,:,k) = Paper_GADFilter_54(x,A,S,G,p,B,C,hparam);

    tt = toc-tt; fprintf('%0.1f s\n',tt);
    tADF(k,1) = tt;

    
    % particle filters
    for j = 1:mN
        
        m = mList(j);
        
        fprintf(['particle filter (m = ' num2str(m) ') ... ']); tt = toc;
        
        zPF(:,:,k,j) = Paper_ParticleFilter_54(m,x,S,A,G,B,C,p,hparam,true);
        
        tt = toc-tt; fprintf('%0.1f s\n',tt);
        tPF(k,j) = tt;
        
    end % j loop over m
    
end % k loop over n


% compute RMSE
r0 = sqrt(mean(mean(bsxfun(@minus,z,0).^2,1),2));
rmu = sqrt(mean(mean(bsxfun(@minus,z,mean(z,2)).^2,1),2));

rf = zeros(nN,1);
rADF = zeros(nN,1);
rDKF = zeros(nN,2);
rPF = zeros(nN,mN);
for k = 1:nN
    rf(k) = sqrt(mean(mean((z-zf(:,:,k)).^2,1),2));
    rADF(k) = sqrt(mean(mean((z-zADF(:,:,k)).^2,1),2));
    rDKF(k,1) = sqrt(mean(mean((z-zDKF(:,:,k,1)).^2,1),2));
    rDKF(k,2) = sqrt(mean(mean((z-zDKF(:,:,k,2)).^2,1),2));
    for j = 1:mN
        rPF(k,j) = sqrt(mean(mean((z-zPF(:,:,k,j)).^2,1),2));
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
end
pl3 = semilogx(nList,rADF,'-','color',[34,139,34]/255,'linewidth',2,'marker','.','markersize',ms); hold on
pl4 = semilogx(nList,rf,':','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl5 = semilogx(nList,rDKF(:,2),'--','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl6 = semilogx(nList,rDKF(:,1),'-','color','r','linewidth',2,'marker','.','markersize',ms); hold on
set(gca,'fontsize',fs,'ylim',[0 1.5])
pl7 = legend([pl3,pl6,pl5,pl4,pl2,pl1],{'G-ADF-UKF','DKF-UKF','robust DKF-UKF','f(X_t)','particle filter','KF, EKF, UKF'});
ylabel('performance (RMSE)'), xlabel('observation dimensionality (n)')
hold off, figure(gcf)

subplot(1,2,2)
fsc = 1; % s per testing dataset
%fsc = 1000/T; % ms per timestep
for k =1:size(tPF,2)
    pl22 = loglog(nList,tPF(:,k)*fsc,'-','color','b','linewidth',1,'marker','.','markersize',ms); hold on
end
pl32 = loglog(nList,tADF*fsc,'-','color',[34,139,34]/255,'linewidth',2,'marker','.','markersize',ms); hold on
pl42 = loglog(nList,tf*fsc,':','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl52 = loglog(nList,(tf+tDKF(:,2))*fsc,'--','color','r','linewidth',2,'marker','.','markersize',ms); hold on
pl62 = loglog(nList,(tf+tDKF(:,1))*fsc,'-','color','r','linewidth',2,'marker','.','markersize',ms); hold on
set(gca,'fontsize',fs,'ylim',[10^-1.2 10^5])
pl72 = legend([pl32,pl62,pl52,pl42,pl22],{'G-ADF-UKF','DKF-UKF','robust DKF-UKF','f(X_t)','particle filter'});
ylabel('computation time (s)'), xlabel('observation dimensionality (n)')
hold off, figure(gcf)

%set(gcf,'paperposition',[0 0 12 5.5])

% record results
save('44.mat');
