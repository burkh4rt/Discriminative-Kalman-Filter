function [z,vars] = KalmanPredict(x,A,C,S,G,mu0,V0)

dz = numel(mu0);
n = size(x,2);
z = zeros(dz,n);

% step 1
K = V0*C.'/(C*V0*C.'+S);
z(:,1) = mu0 + K*(x(:,1)-C*mu0);
V = (eye(dz)-K*C)*V0;
vars = zeros(dz,dz,n);
vars(:,:,1) = V;
% remaining steps
for k = 2:n
    [z(:,k),V] = KalmanOneStep(x(:,k),z(:,k-1),V,A,C,S,G);
    vars(:,:,k) = V;
end
end

function [mu,V] = KalmanOneStep(x,mu,V,A,C,S,G)

P = A*V*A.'+G;

PC = P*C.';
K0 = PC/(C*PC+S);
Amu = A*mu;
mu = Amu+K0*(x-C*Amu);

V = P-K0*C*P;
end