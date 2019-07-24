function [mu,Sig,logp] = BernMixMarg(x,d,p,c,D,a,b,Zmu,Zsig2)
% function [mu,Sig,logp] = BernMixMarg(x,d,p,c,D,a,b,Zmu=0,Zsig2=1)
%
% Consider the model
%
% P(k,L,Z) = a(k,L), if Z(D(k)) < c(k), or
% P(k,L,Z) = b(k,L), if Z(D(k)) >= c(k)
%
% X(k) = rand < P(k,L,Z)
%
% where
% a and b are n x R
% X and c and D are n x 1
% Z is d x 1 multivariate normal with mean 0 and covariance eye(d)
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
% of X at x(:,t).
%
% Details are in NewStuff3.pdf
%
% if p is R x d, then each Z(j) has its own independent L with pmf p(:,j)
% and any coordinate of X with D=j uses that L
%
% if Zmu and or Zsig2 are supplied, then Z has mean Zmu and diagonal
% covariance with Zsig2 on the diagonal

if nargin < 8 || isempty(Zmu), Zmu = zeros(d,1); end
if numel(Zmu) == 1, Zmu = repmat(Zmu,d,1); end

if nargin < 9 || isempty(Zsig2), Zsig2 = ones(d,1); end
if numel(Zsig2) == 1, Zsig2 = repmat(Zsig2,d,1); end

%Zsig2 = max(Zsig2,1e-6);

[n,T] = size(x);
R = size(a,2);
if numel(p) ~= R
    if isequal(size(p),[R,d])
        % independent case
        mu = zeros(d,T);
        Sig = zeros(d,d,T);
        logp = zeros(T,1);
        for j = 1:d
            Dj = D==j;
            [mu(j,:),Sig(j,j,:),logpj] = BernMixMarg(x(Dj,:),1,p(:,j),c(Dj),ones(sum(Dj),1),a(Dj,:),b(Dj,:),Zmu(j),Zsig2(j));
            logp = logp + logpj;
        end
        return
    else
        error('p is incorrectly sized')
    end
end

% transform c
Zsig = sqrt(Zsig2);
c = (c - Zmu(D))./Zsig(D);

cs = sort(c(:));

css = [-realmax;cs;realmax];
psi0 = diff(normcdf(css));
logpsi0 = log(psi0);
psi1 = -diff(normpdf(css));
psi2 = psi0-diff(css.*normpdf(css));

loga = log(a);
log1a = log(1-a);
logb = log(b);
log1b = log(1-b);

pLX = zeros(T,R);
muXL1 = zeros(d,T,R);
muXL2 = zeros(d,T,R);

for L = 1:R
    phi = zeros(d,n+1,T);
    ax = x.*loga(:,L)+(1-x).*log1a(:,L);
    bx = x.*logb(:,L)+(1-x).*log1b(:,L);
    for j = 1:d
        N = D == j;
        for k = 1:n+1
            B = c < css(k+1);
            phi(j,k,:) = sum(ax(N&~B,:),1)+sum(bx(N&B,:),1);
        end
    end
    
    logs0 = phi + reshape(logpsi0,[1,n+1,1]);
    logs0m = max(logs0,[],2);
    pLX(:,L) = sum(log(sum(exp(logs0-logs0m),2))+logs0m,1);
    
    phi = exp(phi - max(phi,[],2));
    
    s0 = reshape(sum(phi.*reshape(psi0,[1,n+1,1]),2),[d,T]);
    s1 = reshape(sum(phi.*reshape(psi1,[1,n+1,1]),2),[d,T]);
    s2 = reshape(sum(phi.*reshape(psi2,[1,n+1,1]),2),[d,T]);

    muXL1(:,:,L) = s1./s0;
    muXL2(:,:,L) = s2./s0;
end

pLX = pLX + reshape(log(p),[1,R]);
mLX = max(pLX,[],2);
pLX = exp(pLX - mLX);
sLX = sum(pLX,2);
pLX = pLX ./ sLX;

logp = mLX+log(sLX);

muXL1p = muXL1 .* reshape(pLX,[1,T,R]);

mu = sum(muXL1p,3);
mu2 = sum(muXL2 .* reshape(pLX,[1,T,R]),3);

Sig = zeros(d,d,T);

diagndx = find(eye(d));

for t = 1:T
    Q = zeros(d,d);
    for L = 1:R
        Q = Q + muXL1p(:,t,L)*muXL1(:,t,L).';
    end
    Q(diagndx) = mu2(:,t);
    Sig(:,:,t) = Q - mu(:,t)*mu(:,t).';
end

% re-transform mu and Sig
mu = Zmu+Zsig.*mu;
Sig = Sig.*reshape(Zsig*Zsig.',[d,d,1]);


