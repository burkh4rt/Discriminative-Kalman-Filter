function [f_est,S_est] = nw_est(ztest,xtrain,ztrain,sz2)
%NW_EST Nadaraya-Watson kernel regression

dz = size(ztest,2);
P_i = (2*pi*sz2)^(-dz/2)*mean(exp(-pdist2(ztrain,ztest,'squaredeuclidean')/2/sz2),2);
sP = sum(P_i,1);
xP = sum(xtrain.*P_i,1);
f_est = xP/sP;
if nargout >1
    [nt, dx] = size(xtrain);
    S_est = zeros(dx,dx);
    for i  =1:nt
        erri = xtrain(i,:)-f_est;
    	S_est = S_est + P_i(i)/sP*(erri'*erri);
    end
end

end

