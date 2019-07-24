function means_filtered = DKF_alt(means, vars, A, G, VZ)
%DKF_FILTERING filters under the DKF model:
% hidden(t)|observed(t) ~ N(means, vars)
% hidden(t)|hidden(t-1) ~ N(A*hidden(t-1), G)
% hidden(t) ~ N(0, VZ)
% this function returns the estimates:
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
    [Vnew,Dnew] = eig(bb,VZ);
    bb = VZ*Vnew*min(Dnew,1)/Vnew;
    Mi = pinv(G+A*S*A');
    S = pinv(pinv(bb)+Mi-pinv(VZ));
    nu = S*(bb\aa+Mi*(A*nu));
    means_filtered(:,t) = nu;
end

end
