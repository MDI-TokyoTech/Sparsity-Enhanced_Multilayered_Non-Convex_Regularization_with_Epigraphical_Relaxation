
function [ xp , zp ] = Proj_Epigraph_L2NU(x, z, w)


[ xp1 , zp1 ] = Proj_Epigraph( reshape( x(:,:,1,:) , [ size(x,1)*size(x,2) , 2 ] ) , reshape( z(:,:,1) , [ size(z,1)*size(z,2) , 1 ] ) , w);
[ xp2 , zp2 ] = Proj_Epigraph( reshape( x(:,:,2:3,:) , [ size(x,1)*size(x,2) , 4 ] ) , reshape( z(:,:,2) , [ size(z,1)*size(z,2) , 1 ] ) , w);

xp = cat( 3, reshape( xp1, [ size(z,1) , size(z,2) , 1 , 2 ] ) , reshape( xp2, [ size(z,1) , size(z,2) , 2 , 2 ] ) );
zp = cat( 3, reshape( zp1, [ size(z,1) , size(z,2) ] ) , reshape( zp2, [ size(z,1) , size(z,2) ] ) );