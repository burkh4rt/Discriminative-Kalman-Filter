function [z,x,w] = Paper_MakeData_52(T,S,A,G,B,C,p)
% function [z,x,w] = Paper_MakeData_52(T,S,A,G,B,C,p)

[n,d,c] = size(B);

z = randn(d,T); 
x = randn(n,T);
w = randsample(c,T,true,p);

G = cholcov(G).';
if size(G,2) < d, G = [G,zeros(d,d-size(G,2))]; end

z(:,1)=cholcov(S).'*z(:,1); 
for t = 2:T 
    z(:,t)=A*z(:,t-1)+G*z(:,t);
end

for k = 1:c, C(:,:,k) = cholcov(C(:,:,k)).'; end
for t = 1:T
    x(:,t) = B(:,:,w(t))*z(:,t)+C(:,:,w(t))*x(:,t);
end
