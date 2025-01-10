function [ x_star , y_star ] = Proj_Epigraph_S2(x, y)
[ U , D , V ] = svd(x,0);
d = diag(D);
d = reshape(d,1,1,length(d(:)));
[ d_star, y_star ] = Proj_Epigraph(d, y);
d_star = d_star(:);
x_star = U*diag(d_star)*V';