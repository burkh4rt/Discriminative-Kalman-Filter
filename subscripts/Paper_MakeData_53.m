function [z,x,w] = Paper_MakeData_53(T,S,A,G,d,p,c,D,a,b,Zmu)
% function [z,x,w] = Paper_MakeData_53(T,S,A,G,d,p,c,D,a,b,Zmu)

if nargin < 11 || isempty(Zmu)
    Zmu = zeros(d,1);
end
Amu = (eye(d)-A)*Zmu;

z = randn(d,T); 

G = cholcov(G).';
if size(G,2) < d, G = [G,zeros(d,d-size(G,2))]; end

z(:,1)=cholcov(S).'*z(:,1)+Zmu; 
for t = 2:T 
    z(:,t)=A*z(:,t-1)+G*z(:,t)+Amu;
end

R = size(a,2);
n = numel(c);

if numel(p) ~= R
    if isequal(size(p),[R,d])
        % independent case (not used in paper)
        c = c(:);
        w = zeros(d,T);
        x = false(n,T);
        for j = 1:d
            w(j,:) = randsample(R,T,true,p(:,j));
            Dj = D==j;
            x(Dj,:) = rand(sum(Dj),T) < (b(Dj,w(j,:)) + (a(Dj,w(j,:))-b(Dj,w(j,:))).*(z(j,:) < c(Dj)));
        end
        return
    else
        error('p is incorrectly sized')
    end
end

w = randsample(R,T,true,p);
x = rand(n,T) < (b(:,w) + (a(:,w)-b(:,w)).*(z(D,:) < c(:)));
