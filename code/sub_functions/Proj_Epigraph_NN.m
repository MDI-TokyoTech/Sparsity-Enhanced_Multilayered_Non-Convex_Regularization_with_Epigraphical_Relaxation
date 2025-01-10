function [ x_star , y_star ] = Proj_Epigraph_NN(x, y)

[ U , D , V ] = svd(x,0);
d = diag(D);
[d_star, y_star] = Proj_Epigraph_L1(d, y);
x_star = U*diag(d_star')*V';