function z = Paper_GADFilter_54(x,AA,M,G,p,H,A,hparam)
% function z = Paper_GADFilter_54(x,A,S,G,p,B,C,hparam)
%
% modified for the code below for the nonlinearity
%
% Implements this code:
%    [z(:,1),Sig] = GaussMixMarg(x(:,1),0,S,p,0,B,C);
%    for t = 2:size(x,2)
%        [z(:,t),Sig] = GaussMixMarg(x(:,t),A*z(:,t-1),A*Sig*A.'+G,p,0,B,C);
%    end

[n,d,R] = size(H);
num = size(x,2);
logp = log(p);

myUKF = unscentedKalmanFilter(@(zz)AA*hfunc(zz,hparam),[],zeros(d,1),...
    'ProcessNoise',G,'StateCovariance',M,'HasAdditiveProcessNoise',true,...
    'Alpha',1,'Beta',0,'Kappa',0);

z = zeros(d,num);
v = zeros(d,1);

for t = 1:num

    xi = zeros(n,R);
    C = zeros(d,n,R);
    GG = zeros(n,n,R);
    CGi = zeros(d,n,R);
    Sigx = zeros(d,d,R);

    pix = zeros(R,1);
    mux = zeros(d,R);

    xibar = zeros(n,1);

    for L = 1:R
        xi(:,L) = H(:,:,L)*v;
        C(:,:,L) = M*H(:,:,L).';
        GG(:,:,L) = H(:,:,L)*C(:,:,L)+A(:,:,L);
        GG(:,:,L) = (GG(:,:,L)+GG(:,:,L).')/2; % force symmetry
        CGi(:,:,L) = C(:,:,L)/GG(:,:,L);
        Sigx(:,:,L) = M - CGi(:,:,L)*C(:,:,L).';
        Sigx(:,:,L) = (Sigx(:,:,L)+Sigx(:,:,L).')/2; % force symmetry
        
        pix(L) = logp(L) + logmvnpdf(x(:,t).',xi(:,L).',GG(:,:,L));
        mux(:,L) = v + CGi(:,:,L)*(x(:,t)-xi(:,L));
        xibar = xibar + p(L)*xi(:,L);
    end

    mp = max(pix);
    pix = exp(pix-mp);
    pix = pix / sum(pix);

    z(:,t) = mux*pix;
    
    %v = AA*z(:,t);
    
    %[v,Dv] = hfunc(z(:,t),hparam);
    %v = AA*v; Dv = AA*Dv;
    
    M = -z(:,t)*z(:,t).';

    for L = 1:R
        M = M + pix(L)*(Sigx(:,:,L)+mux(:,L)*mux(:,L).');
    end   

    myUKF.StateCovariance = M;
    myUKF.State = z(:,t);
    [v,M] = predict(myUKF);
    
    
    %M = AA*M*AA.'+G;
    %M = Dv*M*Dv.'+G;
    
end



