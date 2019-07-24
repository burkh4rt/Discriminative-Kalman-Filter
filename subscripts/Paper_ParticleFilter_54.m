function [z,y,wt] = Paper_ParticleFilter_54(m,x,S,A,G,B,C,p,hparam,fastFlag)
% function [z,y,wt] = Paper_ParticleFilter_54(m,x,S,A,G,B,C,p,hparam,fastFlag=true)

if nargin < 10 || isempty(fastFlag), fastFlag = true; end

[n,d,c] = size(B);
T = size(x,2);
G = cholcov(G).';
if size(G,2) < d, G = [G,zeros(d,d-size(G,2))]; end

z = zeros(d,T);
y = cholcov(S).'*randn(d,m);


if fastFlag 

    const = zeros(c,1);
    for k = 1:c
        Ck = cholcov(C(:,:,k)).';
        const(k) = log(p(k))-(n/2)*log(2*pi)-sum(log(diag(Ck)));
        C(:,:,k) = inv(Ck);
    end
    
    for t = 1:T
        logw = zeros(c,m);
        for k = 1:c
            logw(k,:) = const(k)-0.5*sum((C(:,:,k)*bsxfun(@minus,x(:,t),B(:,:,k)*y)).^2,1);
        end
        
        wt = max(logw,[],1);
        wt = wt + log(sum(exp(bsxfun(@minus,logw,wt)),1));
        wt = exp(wt-max(wt));
        wt = wt/sum(wt);

        z(:,t) = y*wt.';
        
        y = A*hfunc(y(:,randsample(m,m,true,wt)),hparam)+G*randn(d,m);
    end
    
else
    
    for t = 1:T
        
        logw = zeros(c,m);
        for k = 1:c
            logw(k,:) = log(p(k))+logmvnpdf(x(:,t).',(B(:,:,k)*y).',C(:,:,k));
        end
        
        wt = max(logw,[],1);
        wt = wt + log(sum(exp(bsxfun(@minus,logw,wt)),1));
        wt = exp(wt-max(wt));
        wt = wt/sum(wt);

        z(:,t) = y*wt.';
        
        y = A*hfunc(y(:,randsample(m,m,true,wt)),hparam)+G*randn(d,m);
    end

end
