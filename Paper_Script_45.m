p = fileparts(mfilename('fullpath'));
cd(p);
addpath(genpath(p));

%% test name
prefix = '3-';
testname = 'flint';

%% setup
% put the rng in the same state
rng(42,'twister');  % for repeatability

% load preprocessed data
load('flint_procd.mat');

dz = 2;

% specify train and test indices
idx0 = 1:5000;
idx1 = 5000+1:6e3;
n0 = length(idx0);
n1 = length(idx1);

zpred = struct([]);
err = struct([]);
rmse = struct([]);
rtime = struct([]);
maae = struct([]);

%% for each dataset, run filtering
for iRun = 1:6
    vel = procd(iRun).velocities;
    spk = procd(iRun).spikes;

    % shift data so that velocity data is paired with correct neural data
    z = vel(2:end,:)';
    x = spk(1:end-1,:)';

    % split into train and test
    z0 = z(:,idx0);
    z1 = z(:,idx1);
    x0 = x(:,idx0);
    x1 = x(:,idx1);

    % constant baseline "no prediction" at mean
    zpred(iRun).zero = z1*0;

    %% learn parameters for Kalman filter
    z01 = z0(:,2:end);
    z00 = z0(:,1:end-1);
    A0 = z01/z00;
    G0 = (z01-A0*z00)*(z01-A0*z00)'/length(idx0);
    C0 = x0/z0;
    S0 = (x0-C0*z0)*(x0-C0*z0)'/length(idx0);

    % run Kalman filter
    tic;
    zpred(iRun).K = KalmanPredict(x1,A0,C0,S0,G0,zeros([dz,1]),cov(z0'));
    rtime(iRun).K = toc;

    %% DKF-NN
    % nn training
    nn = feedforwardnet(10,'trainbr');
    nn = configure(nn,x0,z0);
    nn = init(nn);
    nn.divideParam.trainRatio = 0.7;
    % trainbr doesn't use validation to prevent overfitting
    % (it uses Bayesian regularization instead)
    nn.divideParam.valRatio = 0;
    nn.divideParam.testRatio = 0.3;
    [nn,tr] = train(nn,x0,z0);

    % estimate error covariance on heldout set
    n01 = length(tr.testInd);
    x01 = x0(:,tr.testInd);
    z01 = z0(:,tr.testInd);
    znn01 = nn(x01);
    z0resnn = z01-znn01;

    z0resOuternn = zeros(dz^2,n01);
    for t = 1:n01, z0resOuternn(:,t) = reshape(z0resnn(:,t)*z0resnn(:,t)',dz^2,1); end

    predsres_loo = @(sx) cell2mat(arrayfun( @(i) nw_est(x01(:,i)',z0resOuternn(:,1:n01~=i)',x01(:,1:n01~=i)',sx)', 1:n01,'UniformOutput',false));
    mseres_loo = @(sx) nanmean(nanmean((z0resOuternn-predsres_loo(sx)).^2));
    sxresoptnn = fminunc(mseres_loo,0.25);

    % predict nn
    tic;
    u_means_nn = zeros([dz,n1]);
    u_vars_nn = zeros([dz,dz,n1]);

    for t = 1:n1
        u_means_nn(:,t) = nn(x1(:,t));
        u_vars_nn(:,:,t) = reshape(nw_est(x1(:,t)',z0resOuternn',x01',sxresoptnn)',[dz,dz]);
    end

    % predict DKF-NN
    zpred(iRun).DKF_NN = DKF_filtering(u_means_nn, u_vars_nn, A0, G0, cov(z0'));
    rtime(iRun).DKF_NN = toc;

    % predict NN-no-filtering
    zpred(iRun).NN_nofiltering = u_means_nn;

    % predict alt DKF-NN
    zpred(iRun).DKF_NN_alt = DKF_alt(u_means_nn, u_vars_nn, A0, G0, cov(z0'));

    % predict robust DKF-NN
    zpred(iRun).DKF_NN_robust = DKF_robust(u_means_nn, u_vars_nn, A0, G0);

    %% DKF-GP
    x00 = x0(:,[tr.trainInd,tr.valInd]);
    z00 = z0(:,[tr.trainInd,tr.valInd]);

    % gp specs
    covfunc = @covSEiso;
    likfunc = @likGauss;

    % gp grid search setup
    l1List = 1:.5:2;
    l2List = -3:.5:-2;
    l3List = -4:.5:-3;

    lZ1 = zeros([length(l1List),length(l2List),length(l3List)]);
    lZ2 = zeros([length(l1List),length(l2List),length(l3List)]);

    % gp grid search to initialize hyperparameter optimization
    for i1 = 1:length(l1List)
        for i2 = 1:length(l2List)
            for i3 = 1:length(l3List)
                lhypi = struct('cov', [l1List(i1); l2List(i2)], 'lik',l3List(i3));
                lZ1(i1,i2,i3) = gp(lhypi,@infExact,[], covfunc, likfunc, x00', z00(1,:)');
                lZ2(i1,i2,i3) = gp(lhypi,@infExact,[], covfunc, likfunc, x00', z00(2,:)');
            end
        end
    end

    [~ ,i1] = min(lZ1(:));
    [i11, i12, i13] = ind2sub(size(lZ1),i1);

    [~ ,i2] = min(lZ2(:));
    [i21, i22, i23] = ind2sub(size(lZ1),i2);

    % train gp hyperparameters from best grid point
    hyp1 = struct('cov',[l1List(i11);l2List(i12)],'lik',l3List(i13));
    hyp2 = struct('cov',[l1List(i21);l2List(i22)],'lik',l3List(i23));
    [hyp1,o1] = minimize(hyp1,@gp,-30,@infExact, [], covfunc, likfunc, x00', z00(1,:)');
    [hyp2,o2] = minimize(hyp2,@gp,-30,@infExact, [], covfunc, likfunc, x00', z00(2,:)');

    % estimate error covariance on heldout set
    zgp01_1 = gp(hyp1,@infExact,[], covfunc, likfunc, x00', z00(1,:)', x01');
    zgp01_2 = gp(hyp2,@infExact,[], covfunc, likfunc, x00', z00(2,:)', x01');
    zgp01 = [zgp01_1,zgp01_2]';
    z0resgp = zgp01 - z01;

    z0resOutergp = zeros(dz^2,n01);
    for t = 1:n01, z0resOutergp(:,t) = reshape(z0resgp(:,t)*z0resgp(:,t)',dz^2,1); end

    predsres_loo = @(sx) cell2mat(arrayfun( @(i) nw_est(x01(:,i)',z0resOutergp(:,1:n01~=i)',x01(:,1:n01~=i)',sx)', 1:n01,'UniformOutput',false));
    mseres_loo = @(sx) nanmean(nanmean((z0resOutergp-predsres_loo(sx)).^2));
    sxresoptgp = fminunc(mseres_loo,0.25);

    % Matlab's builtin gp method provides faster predictions by treating GP
    % as an object instead of as a function that needs to be recomputed
    % for every set of predictions
    gpr1 = fitrgp(x00', z00(1,:)','KernelFunction','squaredexponential',...
        'KernelParameters',exp(hyp1.cov)','Sigma',exp(hyp1.lik),...
        'FitMethod','none');
    gpr2 = fitrgp(x00', z00(2,:)','KernelFunction','squaredexponential',...
        'KernelParameters',exp(hyp2.cov)','Sigma',exp(hyp2.lik),...
        'FitMethod','none');

    % predict gp
    tic;

    % unfiltered gp
    u_means_gp = zeros([dz,n1]);
    u_vars_gp = zeros([dz,dz,n1]);

    for t = 1:n1
        u_means_gp(:,t) = [predict(gpr1,x1(:,t)'), predict(gpr2,x1(:,t)')];
        u_vars_gp(:,:,t) = reshape(nw_est(x1(:,t)',z0resOutergp',x01',sxresoptgp)',[dz,dz]);
    end

    % predict DKF-GP
    zpred(iRun).DKF_GP = DKF_filtering(u_means_gp, u_vars_gp, A0, G0, cov(z0'));
    rtime(iRun).DKF_GP = toc;

    % predict GP-no-filtering
    zpred(iRun).GP_nofiltering = u_means_gp;

    % predict alt DKF-GP
    zpred(iRun).DKF_GP_alt = DKF_alt(u_means_gp, u_vars_gp, A0, G0, cov(z0'));

    % predict robust DKF-GP
    zpred(iRun).DKF_GP_robust = DKF_robust(u_means_gp, u_vars_gp, A0, G0);

    %% DKF-NW
    % optimize sx to minimize leave-one-out MSE on training data
    n00 = n0 - n01;
    predsz_loo = @(sx) cell2mat(arrayfun( @(i) nw_est(x00(:,i)',z00(:,1:n00~=i)',x00(:,1:n00~=i)',sx)', 1:n00,'UniformOutput',false));
    msez_loo = @(sx) nanmean(nanmean((z00-predsz_loo(sx)).^2));
    sxopt = fminunc(msez_loo,0.25);

    z0resnw = cell2mat(arrayfun( @(i) nw_est(x01(:,i)',z00',x00',sxopt),1:n01, 'UniformOutput',false)')'-z01;
    z0resOuternw = zeros(dz^2,n01);
    for t = 1:n01, z0resOuternw(:,t) = reshape(z0resnw(:,t)*z0resnw(:,t)',dz^2,1); end

    predsres_loo = @(sx) cell2mat(arrayfun( @(i) nw_est(x01(:,i)',z0resOuternw(:,1:n01~=i)',x01(:,1:n01~=i)',sx)', 1:n01,'UniformOutput',false));
    mseres_loo = @(sx) nanmean(nanmean((z0resOuternw-predsres_loo(sx)).^2));
    sxresoptnw = fminunc(mseres_loo,5);

    % predict nw
    tic;
    u_means_nw = zeros([dz,n1]);
    u_vars_nw = zeros([dz,dz,n1]);

    for t = 1:n1
        u_means_nw(:,t)  = nw_est(x1(:,t)',z0',x0',sxopt);
        u_vars_nw(:,:,t) = reshape(nw_est(x1(:,t)',z0resOuternw',x01',sxresoptnw)',[dz,dz]);
    end

    for t = 1:n1
        u_means_nw(:,t)  = nw_est(x1(:,t)',z0',x0',sxopt);
        u_vars_nw(:,:,t) = reshape(nw_est(x1(:,t)',z0resOuternw',x01',sxresoptnw)',[dz,dz]);
    end

    % predict DKF-NW
    zpred(iRun).DKF_NW = DKF_filtering(u_means_nw, u_vars_nw, A0, G0, cov(z0'));
    rtime(iRun).DKF_NW = toc;

    % predict NW-no-filtering
    zpred(iRun).NW_nofiltering = u_means_nw;

    % predict alt DKF-NN
    zpred(iRun).DKF_NW_alt = DKF_alt(u_means_nw, u_vars_nw, A0, G0, cov(z0'));

    % predict robust DKF-NN
    zpred(iRun).DKF_NW_robust = DKF_robust(u_means_nw, u_vars_nw, A0, G0);

    % train UKF model
    nnUKF = feedforwardnet(10,'trainbr');
    nnUKF = configure(nnUKF,z0,x0);
    nnUKF = init(nnUKF);
    nn.divideParam.trainRatio = 0.7;
    nn.divideParam.valRatio = 0;
    nn.divideParam.testRatio = 0.3;

    [nnUKF,trUKF] = train(nnUKF,z0,x0);

    xnn0 = nnUKF(z0(:,trUKF.testInd));
    errEstUKF = x0(:,trUKF.testInd)-xnn0;
    covEstUKF = cov(errEstUKF');

    % predict EKF
    tic;
    EKFobj = extendedKalmanFilter(@(z) A0*z, @(z) nnUKF(z), zeros([dz,1]),...
        'ProcessNoise',G0, 'MeasurementNoise',covEstUKF, 'StateCovariance', cov(z0'), 'StateTransitionJacobianFcn', @(z) A0);
    zEKF = zeros([dz,n1]);
    for i = 1:n1
        zEKF(:,i) = correct(EKFobj,x1(:,i));
        predict(EKFobj);
    end
    zpred(iRun).EKF = zEKF;
    rtime(iRun).EKF = toc;

    % predict UKF
    tic;
    UKFobj = unscentedKalmanFilter(@(z) A0*z, @(z) nnUKF(z), zeros([dz,1]),...
        'ProcessNoise',G0, 'MeasurementNoise',covEstUKF, 'StateCovariance', cov(z0'));
    zUKF = zeros([dz,n1]);
    for i = 1:n1
        zUKF(:,i) = correct(UKFobj,x1(:,i));
        predict(UKFobj);
    end
    zpred(iRun).UKF = zUKF;
    rtime(iRun).UKF = toc;


    %% calculate and report performance
    raw2abs_ang_err = @(ae) min(abs([ae-2*pi,ae,ae+2*pi]));

    fnames = fieldnames(zpred);
    for nf = 1:numel(fnames)
        zF = zpred(iRun).(fnames{nf});
        err = zF - z1;
        rmse(iRun).(fnames{nf}) = sqrt(mean(err(:).^2));
        raw_ang_err = atan2(zF(2,:),zF(1,:))-atan2(z1(2,:),z1(1,:));
        abs_ang_err = arrayfun(raw2abs_ang_err,raw_ang_err);
        maae(iRun).(fnames{nf}) = mean(abs_ang_err);
    end

    % remove ancillary variables and save workspace
    clear gpr1 gpr2 x z spk vel i1 i11 i12 i13 i2 i21 i22 i23 i3 nf
    clear l1list l2list l3list zF err
    save(['45_run',num2str(iRun),'.mat']);

end

writetable(struct2table(rmse), [testname,'_rmse.csv'])
writetable(struct2table(rtime), [testname,'_rtime.csv'])
writetable(struct2table(maae), [testname,'_maae.csv'])
