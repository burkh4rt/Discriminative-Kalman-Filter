function [z,f,Q,pdFlag,normFlag] = Paper_DKF_Block_54(x,S,A,G,f,Q,hparam,RobustFlag,DoNorm)
% function [z,f,Q,pdFlag,normFlag] = Paper_DKF_Block_54(x,S,A,G,f,Q,hparam,RobustFlag,DoNorm)
%
% modified for the nonlinear example
%
% runs the DKF iteration when f and Q have already been computed
% x is n x T ... x(:,t) is the observation at time t 
% z is d x T ... z(:,t) is the DKF approximation of the posterior mean
%   given the observations up to time t, namely,
%   z(:,t) = E[Z(:,t)|X(:,1:t)=x(:,1:t)]
% S is the d x d marginal covariance of Z(:,t)
% A is the d x d matrix satisfying E[Z(:,t)|Z(:,t-1)] = AZ(:,t-1)
% G is the d x d conditional covariance of Z(:,t) given Z(:,t-1)
% f is d x T ... f(:,t) is the posterior mean given the observation at 
%   time t, namely,
%   f(:,t) = E[Z(:,t)|X(:,t)=x(:,t)]
% Q is d x d x T ... Q(:,:,t) is the posterior covariance given the
%   observation at time t, namely,
%   Q(:,:,t) = V[Z(:,t)|X(:,t)=x(:,t)]
% RobustFlag = {true,false(default)} ... when true, uses the robust DKF
%
% pdFlag(t) is true if Q(:,:,t) was modified to ensure that S-Q(:,:,t)
%   was positive semidefinite

if nargin < 9 || isempty(DoNorm)
    DoNorm = false;
else
    DoNorm = logical(DoNorm);
end

if nargin < 8 || isempty(RobustFlag)
    RobustFlag = false;
else
    RobustFlag = logical(RobustFlag);
end

d = size(S,1);
T = size(x,2);

z = zeros(d,T);
pdFlag = zeros(1,T);
normFlag = ones(1,T);

Si = inv(S);

% initialize t = 1 with f and Q
mu = f(:,1);
sig = Q(:,:,1);
z(:,1) = mu;

myUKF = unscentedKalmanFilter(@(zz)A*hfunc(zz,hparam),[],mu,...
    'ProcessNoise',G,'StateCovariance',S,'HasAdditiveProcessNoise',true,...
    'Alpha',1,'Beta',0,'Kappa',0);

% loop over the remaining T
for t = 2:T

    % UKF
    myUKF.StateCovariance = sig;
    myUKF.State = mu;
    [Amu,M] = predict(myUKF);
    
    % EKF
    %[hmu,Dmu] = hfunc(mu,hparam);
    %Amu = A*hmu;
    %AA = A*Dmu;
    %M = AA*sig*AA.'+G;
    
    Mi = inv(M);

    if RobustFlag
        Qi = inv(Q(:,:,t));
        sig = inv(Qi+Mi);
    else
        [VV,EE] = eig(Q(:,:,t),S);
        Qi = inv(S*VV*min(EE,1)/VV); 
        sig = inv(Qi-Si+Mi);

        pdFlag(t) = max(diag(EE))-1;
    end
    
    % EKF
    %mu = sig*(Qi*f(:,t)+Mi*AA*mu);
    
    % UKF
    mu = sig*(Qi*f(:,t)+Mi*Amu);
    
    normFlag(t) = sqrt((Amu.'*Amu+f(:,t).'*f(:,t))/(mu.'*mu+eps(0)));
    if DoNorm, mu = mu .* min(1,normFlag(t)); end % not used in paper (useful for EKF)
    
    z(:,t) = mu;
   
end
