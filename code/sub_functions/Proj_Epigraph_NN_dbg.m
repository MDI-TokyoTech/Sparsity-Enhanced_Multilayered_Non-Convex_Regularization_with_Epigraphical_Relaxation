function [ x_star , y_star ] = Proj_Epigraph_NN_dbg(x, y, bs)

if ~exist('bs','var')
    bs = [ 1 2 ];
end

% [ U , D , V ] = svd(x,0);
% d = diag(D); d=d';
% dy_star = Proj_Epigraph_L1(d, y);
% d_star = dy_star(1,1:size(d,2));
% y_star = dy_star(1,end);
% x_star = U*diag(d_star')*V';

x1 = x(1:9,1:2); x2 = x(10:18,3:4); x3 = x(19:27,5:6);
y1 = y(1); y2 = y(2); y3 = y(3);
[ U1 , D1 , V1 ] = svd(x1,0);
d1 = reshape( diag( D1 ) , [1,2] );
dy_star1 = Proj_Epigraph_L1(d1, y1);
d_star1 = dy_star1( : , 1:end-1 );
y_star1 = dy_star1( : , end );
x_star1 = U1*diag( reshape( d_star1 , size(diag(D1)) ) )*V1';
fprintf( 'Error x: %f, Error y: %f\n' , norm( x_star1 - x1 ) , norm( y_star1 - y1 ) )

[ U2 , D2 , V2 ] = svd(x2,0);
d2 = reshape( diag( D2 ) , [1,2] );
dy_star2 = Proj_Epigraph_L1(d2, y2);
d_star2 = dy_star2( : , 1:end-1 );
y_star2 = dy_star2( : , end );
x_star2 = U2*diag( reshape( d_star2 , size(diag(D2)) ) )*V2';
fprintf( 'Error x: %f, Error y: %f\n' , norm( x_star2 - x2 ) , norm( y_star2 - y2 ) )

[ U3 , D3 , V3 ] = svd(x3,0);
d3 = reshape( diag( D3 ) , [1,2] );
dy_star3 = Proj_Epigraph_L1(d3, y3);
d_star3 = dy_star3( : , 1:end-1 );
y_star3 = dy_star3( : , end );
x_star3 = U3*diag( reshape( d_star3 , size(diag(D3)) ) )*V3';
fprintf( 'Error x: %f, Error y: %f\n' , norm( x_star3 - x3 ) , norm( y_star3 - y3 ) )


[ U , D , V ] = svd(x,0);
d = reshape( diag( D ) , bs );
dy_star = Proj_Epigraph_L1(d, y);
d_star = dy_star( : , 1:end-1 );
y_star = dy_star( : , end );
x_star = U*diag( reshape( d_star , size(diag(D)) ) )*V';
fprintf( 'Error x: %f, Error y: %f\n' , norm( x_star - x ) , norm( y_star - y ) )
test = 1;