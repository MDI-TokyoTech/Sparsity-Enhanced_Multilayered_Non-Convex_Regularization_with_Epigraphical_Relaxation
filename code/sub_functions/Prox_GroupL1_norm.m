function [Y] = Prox_GroupL1_norm(X, blocksize, gamma)

Du = zeros( size(X,1)/blocksize(1) , size(X,2)/blocksize(2) , blocksize(1)*blocksize(2) ); 

p = 1;
for n1 = 1:blocksize(1)
    for n2 = 1:blocksize(2)
        Du(:,:,p) = X(n1:blocksize(1):end,n2:blocksize(2):end);
        p = p + 1;
    end
end

n = size(Du);
onemat = ones(n(1:2));
thresh = ((sqrt(sum(Du.^2, 3))).^(-1))*gamma;
thresh(thresh > 1) = 1;
coef = (onemat - thresh);

for k = 1:n(3)
        Du(:,:,k) = coef.*Du(:,:,k);
end

Y = zeros( size(X) );
p = 1;
for n1 = 1:blocksize(1)
    for n2 = 1:blocksize(2)
        Y(n1:blocksize(1):end,n2:blocksize(2):end) = Du(:,:,p);
        p = p + 1;
    end
end

