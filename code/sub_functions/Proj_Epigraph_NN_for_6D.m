function [ PDu , y ] = Proj_Epigraph_NN_for_6D(PDu, y, blocksize)

[v, h, c, d, s1, s2] = size(PDu);

for i = 1:floor(v/blocksize(1)) % rows of the image
    for j = 1:floor(h/blocksize(2)) % cols of the image
        for k = 1:s1 % rows of each karnel
            for l = 1:s2 % cols of each karnel
                % extract pixels
                block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, : , : , k , l);
                M = reshape(block,[prod(blocksize),c*d]); % difference vectors of each color

                % projection to epigraph of Nuclear Norm
                [ Mprj , yprj ] = Proj_Epigraph_NN( M, y(i,j,1,1,k,l) );

                % assign results
                PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,:,:,k,l) = reshape(Mprj, [blocksize, c, d]); 
                y(i,j,1,1,k,l) = yprj;
            end
        end
    end
end