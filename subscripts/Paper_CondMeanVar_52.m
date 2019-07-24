function [f,Q,e] = Paper_CondMeanVar_52(x,S,B,C,p)
% function [f,Q,e] = Paper_CondMeanVar_52(x,S,B,C,p)
%
% Consider the model
% X = B(:,:,L)*Z + sqrtm(C(:,:,L))*randn(n,n)
% where
% X is n x 1
% Z is d x 1
% B is n x d x k
% C is n x n x k
% and L is a randomly chosen interger from 1:k using the pmf in the L-vector p 
%
% The function computes the conditional mean and covariance of Z given X, 
% namely,
%
% x is n x T
% f is d x T
% Q is d x d x T
%
% f(:,t) = E[Z|X=x(:,t)]
% Q(:,:,t) = V[Z|X=x(:,t)]
%
% It also computes e(t)=P(x(:,t)), meaning the probability density of X at
% x(:,t).

%
[~,d,c] = size(B);
T = size(x,2);
Si = inv(S); Si = 0.5*(Si+Si.');

D = zeros(d,d,c);
Vx = zeros(d,T,c);
wt = zeros(1,T,c);
for k = 1:c
    BCk = B(:,:,k).'/C(:,:,k);
    Dki = Si+BCk*B(:,:,k);  
    D(:,:,k) = inv(Dki);
    Vx(:,:,k) = Dki\BCk*x;

    wt(1,:,k) = log(p(k)) + logmvnpdf(x.',[],C(:,:,k)+B(:,:,k)*S*B(:,:,k).');
end

e = max(wt,[],3);
e = e + log(sum(exp(bsxfun(@minus,wt,e)),3));

wt = exp(bsxfun(@minus,wt,e));

f = sum(bsxfun(@times,Vx,wt),3);

Q = zeros(d,d,T);
for t = 1:T
    QQ = -f(:,t)*f(:,t).';
    for k = 1:c
        QQ = QQ+wt(1,t,k)*(Vx(:,t,k)*Vx(:,t,k).'+D(:,:,k));
    end
    Q(:,:,t) = 0.5*(QQ+QQ.'); % remove asymmetries from rounding errors
end

e = exp(e);
