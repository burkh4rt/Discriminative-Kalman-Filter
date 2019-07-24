function [z,y,wt] = Paper_ParticleFilter_53(m,x,S,A,G,d,p,c,D,a,b,muZ)
% function [z,y,wt] = Paper_ParticleFilter_53(m,x,S,A,G,d,p,c,D,a,b,muZ)

if nargin < 12 || isempty(muZ)
    muZ = zeros(d,1);
end
muA = (eye(d)-A)*muZ;

[~,T] = size(x);
R = size(a,2);
c = c(:);

indflag = false;
if numel(p) ~= R
    if isequal(size(p),[R,d])
        % independent case (not used in paper)
        indflag = true;
    else
        error('p is incorrectly sized')
    end
end

G = cholcov(G).';
if size(G,2) < d, G = [G,zeros(d,d-size(G,2))]; end

z = zeros(d,T);
y = cholcov(S).'*randn(d,m)+muZ;

ab = a-b;

for t = 1:T
    
    if indflag % not used in paper
        
        wt = zeros(1,m);
        for j = 1:d
            logw = zeros(R,m);
            Dj = D == j;
            if any(Dj)
                for k = 1:R
                    pk = b(Dj,k) + ab(Dj,k).*(y(j,:) < c(Dj));
                    logw(k,:) = log(p(k,j))+sum(x(Dj,t).*log(pk)+(1-x(Dj,t)).*log(1-pk),1);
                end
            end
            wtj = max(logw,[],1);
            wt = wt + wtj + log(sum(exp(bsxfun(@minus,logw,wtj)),1));
        end
        
    else
        
        logw = zeros(R,m);

        for k = 1:R
            pk = b(:,k) + ab(:,k).*(y(D,:) < c);
            logw(k,:) = log(p(k))+sum(x(:,t).*log(pk)+(1-x(:,t)).*log(1-pk),1);
        end
        
        wt = max(logw,[],1);
        wt = wt + log(sum(exp(bsxfun(@minus,logw,wt)),1));

    end
    
    wt = exp(wt-max(wt));
    wt = wt/sum(wt);
        
    z(:,t) = y*wt.';
    y = A*y(:,randsample(m,m,true,wt))+G*randn(d,m)+muA;
end

