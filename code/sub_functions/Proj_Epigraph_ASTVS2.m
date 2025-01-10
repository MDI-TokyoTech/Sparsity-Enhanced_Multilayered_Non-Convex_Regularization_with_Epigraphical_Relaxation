function [ PDu , y ] = Proj_Epigraph_ASTVS2(PDu, y, blocksize)

[v, h, c, d, s1, s2] = size(PDu);

for i = 1:floor(v/blocksize(1))
    for j = 1:floor(h/blocksize(2))
        for k = 1:s1
            for l = 1:s2
                block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, : , : , k , l);
                M = reshape(block,[prod(blocksize),c*d]);
                [ Mprj, yprj ] = Proj_Epigraph_S2( M , y(i,j,1,1,k,l) );
                PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,:,:,k,l) = reshape(Mprj, [blocksize, c, d]); 
                y(i,j,1,1,k,l) = yprj;
            end
        end
    end
end