function [ PDu , y ] = Proj_Epigraph_cwSTV(PDu, y, blocksize)
% original
[v, h, c, d, s1, s2] = size(PDu);
for i = 1:floor(v/blocksize(1))
    for j = 1:floor(h/blocksize(2))
        for k = 1:s1
            for l = 1:s2
                for col = 1:c
                    block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, col , : , k , l);
                    M = reshape(block,[prod(blocksize),d]);
                    [ Mprj, yprj ] = Proj_Epigraph_NN( M, y(i,j,col,1,k,l) );
                    PDu(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,col,:,k,l) = reshape(Mprj, [blocksize, 1, d]);
                    y(i,j,col,1,k,l) = yprj;
                end
            end
        end
    end
end
% --------------------------------------

% [v, h, c, d, s1, s2] = size(PDu);
% PDu2 = zeros(size(PDu));
% y2 = zeros(size(y));
% for i = 1:floor(v/blocksize(1))
%     for j = 1:floor(h/blocksize(2))
%         for k = 1:s1
%             for l = 1:s2
%                 block = PDu(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, : , : , k , l);
%                 M1 = reshape( block(:,:,1,:) , [ prod(blocksize), d ]);
%                 M2 = reshape( block(:,:,2,:) , [ prod(blocksize), d ]);
%                 M3 = reshape( block(:,:,3,:) , [ prod(blocksize), d ]);
%                 
%                 [ Mprj1, yprj1 ] = Proj_Epigraph_NN( M1, y(i,j,1,1,k,l) );
%                 [ Mprj2, yprj2 ] = Proj_Epigraph_NN( M2, y(i,j,2,1,k,l) );
%                 [ Mprj3, yprj3 ] = Proj_Epigraph_NN( M3, y(i,j,3,1,k,l) );
%                                 
%                 PDu2(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,1,:,k,l) = reshape(Mprj1, [blocksize, 1, d]);
%                 PDu2(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,2,:,k,l) = reshape(Mprj2, [blocksize, 1, d]);
%                 PDu2(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,3,:,k,l) = reshape(Mprj3, [blocksize, 1, d]);
%                 
%                 y2(i,j,1,1,k,l) = yprj1;
%                 y2(i,j,2,1,k,l) = yprj2;
%                 y2(i,j,3,1,k,l) = yprj3;
%             end
%         end
%     end
% end
