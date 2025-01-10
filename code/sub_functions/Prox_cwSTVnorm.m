function[PDu] = Prox_cwSTVnorm(PDu, gamma, blocksize)
% channel-wise STV

[v, h, c, d, s1, s2] = size(PDu);

for i = 1:v/blocksize(1)
    for j = 1:h/blocksize(2)
        for k = 1:s1
            for l = 1:s2
                for col = 1:c
                    block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, col , : , k , l);
                    M = reshape(block,[prod(blocksize),d]);
                    [U, S, V] = svd(M,0);
                    Sthre = diag(max(0, diag(S) - gamma));
                    Sthre = reshape( Sthre , size(S) );
                    PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,col,:,k,l)...
                        = reshape(U*Sthre*V', [blocksize, 1, d]);
                end
            end
        end
    end
end
