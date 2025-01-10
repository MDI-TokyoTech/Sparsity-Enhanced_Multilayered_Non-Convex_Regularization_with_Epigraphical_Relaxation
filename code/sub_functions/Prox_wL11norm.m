% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function[Du] = Prox_wL11norm(Du, gamma, w)

Du(:,:,1) = Prox_L1norm( Du(:,:,1) , gamma*w );
Du(:,:,2) = Prox_L1norm( Du(:,:,2) , gamma );









