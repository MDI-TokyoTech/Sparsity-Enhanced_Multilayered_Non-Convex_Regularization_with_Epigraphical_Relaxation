function[PDu] = Prox_MSTVnorm(PDu, gamma, blocksize)
[v, h, c, d, s1, s2] = size(PDu);
M = zeros(prod(blocksize), c*d);

for i = 1:v/blocksize(1)
    for j = 1:h/blocksize(2)
        for k = 1:s1
            for l = 1:s2
                a = 1;
                for q = 1:d % d before c is for computing reshape()
                    for p = 1:c
                        block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, p ,q, k, l);
                        M(:,a) = block(:);
                        a = a+1;
                    end
                end
                [U, S, V] = svd(M,0);
                Sthre_ = diag(max(0, diag(S) - gamma));
                Sthre = zeros(size(S));
                Sthre(1:size(Sthre_,1),1:size(Sthre_,2)) = Sthre_;
                PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,:,:,k,l)...
                    = reshape(U*Sthre*V', [blocksize, c, d]);                
            end
        end
    end
end
