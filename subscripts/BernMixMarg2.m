function [mu,Sig,logp] = BernMixMarg2(x,d,p,c,D,a,b)
% function [mu,Sig,logp] = BernMixMarg2(x,d,p,c,D,a,b)
%
% Uses the calculations in the appendix of the paper
% designed to test BernMixMarg and eqn in text


[n,T] = size(x);
L = size(a,2);

nk = zeros(d,1);
for k = 1:d
    nk(k) = sum(D==k);
end
nmax = max(nk);

N = zeros(nmax,d);
C = repmat(realmax,nmax+2,d);
C(1,:) = -realmax;
nCk = zeros(d,1);
for k = 1:d
    if nk(k) > 0
        ntmp = find(D==k);
        ctmp = unique(c(ntmp));
        nCk(k) = numel(ctmp);
        N(1:nk(k),k) = ntmp;
        C(2:nCk(k)+1,k) = ctmp;
    end
end

phi1 = zeros(nmax+1,d);
phi2 = zeros(nmax+1,d);
phi3 = zeros(nmax+1,d);
for k = 1:d
    for j = 1:nCk(k)+1
        phi1(j,k) = normcdf(C(j+1,k))-normcdf(C(j,k));
        phi2(j,k) = normpdf(C(j,k))-normpdf(C(j+1,k))-phi1(j,k);
        phi3(j,k) = C(j,k)*normpdf(C(j,k))-C(j+1,k)*normpdf(C(j+1,k))-2*phi2(j,k);
    end
end

gam = zeros(n,nmax+1,L);
for l = 1:L
    for i = 1:n
        k = D(i);
        ai = a(i,l);
        bi = b(i,l);
        ci = c(i);
        for j = 1:nCk(k)+1
            gam(i,j,l) = bi+(ai-bi)*(C(j+1,k) <= ci);
        end
    end
end
gam1 = 1-gam;

mu = zeros(d,T);
Sig = zeros(d,d,T);
logp = zeros(T,1);

for t = 1:T
    
    DD = zeros(nmax+1,d,L);
    mx = -realmax;
    for l = 1:L
        for k = 1:d
            m = nCk(k)+1;
            DD(1:m,k,l) = sum(log(gam(N(1:nk(k),k),1:m,l).*x(N(1:nk(k),k),t)+...
                gam1(N(1:nk(k),k),1:m,l).*~x(N(1:nk(k),k),t)),1);
            mx = max(mx,max(DD(1:m,k,l)));
        end
    end
    DD = exp(DD-mx);
    
    EE = squeeze(prod(sum(DD.*phi1,1),2));
    AA = sum(EE.*p);
    
    logp(t) = mx+log(AA);
    
    BB = zeros(d,1);
    for k = 1:d
        BB(k) = sum(EE.*squeeze(sum(DD(:,k,:).*(phi1(:,k)+phi2(:,k)),1)./sum(DD(:,k,:).*phi1(:,k),1)).*p);
    end
    CC = zeros(d,d);
    for k = 1:d
        for kk = 1:k-1
            CC(k,kk) = sum(EE.*squeeze(sum(DD(:,k,:).*(phi1(:,k)+phi2(:,k)),1).*sum(DD(:,kk,:).*(phi1(:,kk)+phi2(:,kk)),1)./(sum(DD(:,k,:).*phi1(:,k),1).*sum(DD(:,kk,:).*phi1(:,kk),1))).*p);
            CC(kk,k) = CC(k,kk);
        end
        CC(k,k) = sum(EE.*squeeze(sum(DD(:,k,:).*(phi1(:,k)+phi2(:,k)+phi2(:,k)+phi3(:,k)),1)./sum(DD(:,k,:).*phi1(:,k),1)).*p);
    end
    
    mu(:,t) = BB/AA;
    Sig(:,:,t) = CC/AA-mu(:,t)*mu(:,t).';
end



