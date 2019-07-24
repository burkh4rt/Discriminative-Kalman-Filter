function z = Paper_GADFilter_52(x,AA,M,G,p,H,A)
% function z = Paper_GADFilter_52(x,A,S,G,p,B,C)
%
% Implements this code:
%    [z(:,1),Sig] = GaussMixMarg(x(:,1),0,S,p,0,B,C);
%    for t = 2:size(x,2)
%        [z(:,t),Sig] = GaussMixMarg(x(:,t),A*z(:,t-1),A*Sig*A.'+G,p,0,B,C);
%    end

[n,d,R] = size(H);
num = size(x,2);
logp = log(p);

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
    v = AA*z(:,t);
    M = -z(:,t)*z(:,t).';

    for L = 1:R
        M = M + pix(L)*(Sigx(:,:,L)+mux(:,L)*mux(:,L).');
    end   
    
    M = AA*M*AA.'+G;
end



