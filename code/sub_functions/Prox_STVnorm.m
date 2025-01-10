function PDu = Prox_STVnorm(PDu, gamma, blocksize)
[v, h, c, d, s1, s2] = size(PDu);

for i = 1:v/blocksize(1)
    for j = 1:h/blocksize(2)
        for k = 1:s1
            for l = 1:s2
                block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, : , : , k , l);
                M = reshape(block,[blocksize(1)*blocksize(2)*c,d]);
                block1 = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, 1 , : , k , l);
                block2 = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, 2 , : , k , l);
                block3 = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, 3 , : , k , l);
                M1 = reshape(block1,[blocksize(1)*blocksize(2),d]);
                M2 = reshape(block2,[blocksize(1)*blocksize(2),d]);
                M3 = reshape(block3,[blocksize(1)*blocksize(2),d]);
                [U, S, V] = svd(M,0);
                Sthre = diag(max(0, diag(S) - gamma));
                PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,:,:,k,l) = reshape(U*Sthre*V', [blocksize, c, d]);
            end
        end
    end
end