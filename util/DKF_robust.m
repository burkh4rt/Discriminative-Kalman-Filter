function means_filtered = DKF_robust(means, vars, A, G)
%DKF_ROBUST filters under the DKF model:
% hidden(t)|observed(t) ~ N(means, vars)
% hidden(t)|hidden(t-1) ~ N(A*hidden(t-1), G)
% hidden(t) ~ N(0, VZ)
% this function returns robust estimates:
% hidden(t)|observed(1:t-1) ~ N(means_filtered, vars_filtered)
% means are [n,T] dimensional
% vars are [n,n,T] dimensional

[n,T] = size(means);
means_filtered = zeros([n,T]);

S = vars(:,:,1);
nu = means(:,1);
means_filtered(:,1) = nu;

for t = 2:T
    aa = means(:,t);
    bb = vars(:,:,t);
    Mi = pinv(G+A*S*A');
    S = pinv(pinv(bb)+Mi);
    nu = S*(bb\aa+Mi*(A*nu));
    means_filtered(:,t) = nu;
end

end
