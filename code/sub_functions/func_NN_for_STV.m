function[PDuNN] = func_NN_for_STV(PDu, blocksize)
[v, h, c, d, s1, s2] = size(PDu);
PDuNN = zeros([floor(v/blocksize(1)),floor(h/blocksize(2)),c,1,s1,s2]);

for i = 1:floor(v/blocksize(1))
    for j = 1:floor(h/blocksize(2))
        for k = 1:s1
            for l = 1:s2
                for col = 1:c
                    block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, col , : , k , l);
                    M = reshape(block,[prod(blocksize),d]);
                    [~, S, ~] = svd(M, "econ");
                    PDuNN(i,j,col,1,k,l) = sum(diag(S));
                end
            end
        end
    end
end