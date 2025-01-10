function [ xp , zp ] = Proj_Epigraph_L2_for_DVTV_GPU( x , z , wlumi )
% Epigraph projection to the L2 norm of luminance alone
% = Epigraph projection onto 'wx = z'
[ xpwx, zpwx ] = Proj_Epigraph_L2_for_DVTV_GPU_lumi( x( :, :, 1, :), z( :, :, 1, :), wlumi);
% L2 norm of color1 and color2
[ xpL2, zpL2 ] = Proj_Epigraph_L2_for_DVTV_GPU_chro( x( : , :, 2:3, :), z( :, :, 2, :), 1);

xp = gpuArray(cat( 3, xpwx , xpL2 ));
zp = gpuArray(cat( 3, zpwx , zpL2 ));