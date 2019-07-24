function [mu,Sig] = GaussMixMarg2(x,v,M,p,b,H,A)
% function [mu,Sig] = GaussMixMarg2(x,v,M,p,b,H,A)
%
% Uses the calculations in the appendix of the paper
% Written to test GaussMixMarg.m using alternative derivations


[n,d,R] = size(H);
num = size(x,2);
if numel(v)==1, v = repmat(v,d,1); end
if numel(b)==1, xi = repmat(b,n,R); else, xi = b; end

U = zeros(d,d,R);
y = zeros(d,num,R);
logKp = zeros(num,R);
logdp = zeros(num,R);

for L = 1:R
    U(:,:,L) = inv(inv(M)+H(:,:,L).'*inv(A(:,:,L))*H(:,:,L));
    U(:,:,L) = (U(:,:,L)+U(:,:,L).')/2;
    for  t = 1:num
        y(:,t,L) = U(:,:,L)*(inv(M)*v+H(:,:,L).'*inv(A(:,:,L))*(x(:,t)-xi(:,L)));
        logKp(t,L) = log(p(L))+logmvnpdf(v.',zeros(1,d),M)+logmvnpdf(x(:,t).',xi(:,L).',0.5*(A(:,:,L)+A(:,:,L).'))-logmvnpdf(y(:,t,L).',zeros(1,d),U(:,:,L));
        logdp(t,L) = log(p(L))+logmvnpdf(x(:,t).',(xi(:,L)+H(:,:,L)*v).',A(:,:,L)+H(:,:,L)*M*H(:,:,L).');
    end
end

mu = zeros(d,num);
Sig = zeros(d,d,num);

for t = 1:num
    mx = max(logdp(t,:));
    mx = mx + log(sum(exp(logdp(t,:)-mx)));
    w = exp(logKp(t,:)-mx);
    for L = 1:R
        mu(:,t) = mu(:,t) + w(L)*y(:,t,L);
        Sig(:,:,t) = Sig(:,:,t) + w(L)*(U(:,:,L)+y(:,t,L)*y(:,t,L).');
    end
    Sig(:,:,t) = Sig(:,:,t) - mu(:,t)*mu(:,t).';
end



