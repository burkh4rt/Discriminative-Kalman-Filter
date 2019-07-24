function [mu,Sig,logp,muT,SigT] = GaussMixMarg(x,v,M,p,b,H,A)
% function [mu,Sig,logp,muT,SigT] = GaussMixMarg(x,v,M,p,b,H,A)
%
% Consider the model
% X = b(:,L) + H(:,:,L)*Z + sqrtm(A(:,:,L))*randn(n,n)
% where
% X is n x 1
% Z is d x 1 multivariate normal with mean v and covariance M
% H is n x d x k
% A is n x n x k
% and L is a randomly chosen interger from 1:R using the pmf in the R-vector p 
%
% The function computes the conditional mean and covariance of Z given X, 
% namely,
%
% x is n x T
% mu is d x T
% Sig is d x d x T
%
% mu(:,t) = E[Z|X=x(:,t)]
% Sig(:,:,t) = V[Z|X=x(:,t)]
%
% It also computes logp(t)=log(P(x(:,t))), meaning the log probability 
% density of X at x(:,t).
%
% Let (XX,ZZ) be a pair of jointly-Gaussian random variables that have the 
% same mean and covariance as (X,Z). It also computes
%
% muT(:,t) = E[ZZ|XX=x(:,t)]
% SigT(:,:,t) = V[ZZ|XX=x(:,t)]
%
% Details are in NewStuff2.pdf


[n,d,R] = size(H);
num = size(x,2);
if numel(v)==1, v = repmat(v,d,1); end
if numel(b)==1, xi = repmat(b,n,R); else, xi = b; end

C = zeros(d,n,R);
G = zeros(n,n,R);
CGi = zeros(d,n,R);
Sigx = zeros(d,d,R);

pix = zeros(R,num);
mux = zeros(d,R,num);

Cbar = zeros(d,n);
Gtilde = zeros(n,n);
xibar = zeros(n,1);

for L = 1:R
    xi(:,L) = xi(:,L)+H(:,:,L)*v;
    C(:,:,L) = M*H(:,:,L).';
    G(:,:,L) = H(:,:,L)*C(:,:,L)+A(:,:,L);
    G(:,:,L) = (G(:,:,L)+G(:,:,L).')/2; % force symmetry
    CGi(:,:,L) = C(:,:,L)/G(:,:,L);
    Sigx(:,:,L) = M - CGi(:,:,L)*C(:,:,L).';
    Sigx(:,:,L) = (Sigx(:,:,L)+Sigx(:,:,L).')/2; % force symmetry
    
    pix(L,:) = log(p(L)) + logmvnpdf(x.',xi(:,L).',G(:,:,L));
    mux(:,L,:) = v + CGi(:,:,L)*(x-xi(:,L));
    
    Cbar = Cbar + p(L)*C(:,:,L);
    Gtilde = Gtilde + p(L)*(G(:,:,L)+xi(:,L)*xi(:,L).');
    xibar = xibar + p(L)*xi(:,L);
end
Gtilde = Gtilde - xibar*xibar.';
CGitilde = Cbar/Gtilde;

mp = max(pix,[],1);
pix = exp(pix-mp); 
logp = sum(pix,1);
pix = pix ./ logp;
logp = logp + mp;

mu = zeros(d,num);
Sig = zeros(d,d,num);
muT = repmat(v,1,num);
SigT = repmat(M-CGitilde*Cbar.',1,1,num);

for k = 1:num
    for L = 1:R
        mu(:,k) = mu(:,k) + pix(L,k)*mux(:,L,k);
        Sig(:,:,k) = Sig(:,:,k) + pix(L,k)*(Sigx(:,:,L)+mux(:,L,k)*mux(:,L,k).');
    end
    Sig(:,:,k) = Sig(:,:,k) - mu(:,k)*mu(:,k).';
    
    muT(:,k) = muT(:,k) + CGitilde*(x(:,k)-xibar);
end



