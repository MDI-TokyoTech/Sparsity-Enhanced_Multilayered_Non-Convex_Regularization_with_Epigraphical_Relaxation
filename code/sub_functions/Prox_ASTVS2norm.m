function[PDu] = Prox_ASTVS2norm(PDu, gamma, blocksize)
[v, h, c, d, s1, s2] = size(PDu);

for i = 1:v/blocksize(1)
    for j = 1:h/blocksize(2)
        for k = 1:s1
            for l = 1:s2
                block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, : , : , k , l);
                M = reshape(block,[prod(blocksize),c*d]);
                M = prox_L2(M, gamma);
                PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,:,:,k,l) = reshape(M, [blocksize, c, d]);
            end
        end
    end
end