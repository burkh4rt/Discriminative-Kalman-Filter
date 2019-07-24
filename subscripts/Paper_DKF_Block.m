function [z,f,Q,pdFlag] = Paper_DKF_Block(x,S,A,G,f,Q,RobustFlag)
% function [z,f,Q,pdFlag] = Paper_DKF_Block(x,S,A,G,f,Q,RobustFlag)
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

if nargin < 7 || isempty(RobustFlag)
    RobustFlag = false;
else
    RobustFlag = logical(RobustFlag);
end

d = size(S,1);
T = size(x,2);

z = zeros(d,T);
pdFlag = zeros(1,T);

Si = inv(S);

% initialize t = 1 with f and Q
mu = f(:,1);
sig = Q(:,:,1);
z(:,1) = mu;

% loop over the remaining T
for t = 2:T

    M = A*sig*A.'+G;
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
    
    mu = sig*(Qi*f(:,t)+Mi*A*mu);
    
    z(:,t) = mu;
   
end
