% Only for 2^n square image

function[x] = func_NoiseletTrans2D(x)
x(:,:) = realnoiselet(x(:,:));
x = x/size(x,1);