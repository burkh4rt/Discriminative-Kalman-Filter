function [z,Dz] = hfunc(z,bc)
% nonlinearity used in PaperScript_54
%
% z(:,t) on the output is the function applied to z(:,t) on the input
% Dz(:,:,t) is the jacobian matrix at z(:,t)

d = size(z,1);
if nargin < 2 || isempty(bc), bc = 1; end

b = sqrt(bc*sum(z.^2,1));
cosb = cos(b);
sinb = sin(b);
zz = z;

for k = 1:2:d-1
    z(k,:) = cosb.*zz(k,:)-sinb.*zz(k+1,:);
    z(k+1,:) = sinb.*zz(k,:)+cosb.*zz(k+1,:);
end

if nargout > 1
    T = size(z,2);
    if T == 1
        Dz = eye(d);
        for k = 1:2:d-1
            Dz(k,:) = -bc*(sinb*zz(k)+cosb*zz(k+1))/b*zz.';
            Dz(k+1,:) = bc*(cosb*zz(k)-sinb*zz(k+1))/b*zz.';
            Dz(k,k) = Dz(k,k)+cosb; Dz(k,k+1) = Dz(k,k+1)-sinb;
            Dz(k+1,k) = Dz(k+1,k)+sinb; Dz(k+1,k+1) = Dz(k+1,k+1)+cosb;
        end
    else
        Dz = zeros(d,d,T);
        for t = 1:T
            Dz(:,:,t) = eye(d); 
            for k = 1:2:d-1
                Dz(k,:,t) = -bc*(sinb(t)*zz(k,t)+cosb(t)*zz(k+1,t))/b(t)*zz(:,t).';
                Dz(k+1,:,t) = bc*(cosb(t)*zz(k,t)-sinb(t)*zz(k+1,t))/b(t)*zz(:,t).';
                Dz(k,k,t) = Dz(k,k,t)+cosb(t); Dz(k,k+1,t) = Dz(k,k+1,t)-sinb(t);
                Dz(k+1,k,t) = Dz(k+1,k,t)+sinb(t); Dz(k+1,k+1,t) = Dz(k+1,k+1,t)+cosb(t);
            end
        end
    end
end


