function [ PDu , y ] = Proj_Epigraph_GSTV(PDu, y, blocksize)
[v, h, c, d, s1, s2] = size(PDu);
for i = 1:floor(v/blocksize(1))
    for j = 1:floor(h/blocksize(2))
        for k = 1:s1
            for l = 1:s2
                for col = 1:c
                    block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, col , : , k , l);
                    M = reshape(block,[prod(blocksize),d]);
                    [ Mprj, yprj ] = Proj_Epigraph_NN( M, y(i,j,col,1,k,l) );
                    PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,col,:,k,l) = reshape(Mprj,[blocksize, 1, d]);
                    y(i,j,col,1,k,l) = yprj;
                end
            end
        end
    end
end