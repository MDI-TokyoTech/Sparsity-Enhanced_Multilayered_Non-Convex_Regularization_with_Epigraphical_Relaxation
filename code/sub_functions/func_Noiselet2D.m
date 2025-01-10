% Only for 2^n square image

function[x] = func_Noiselet2D(x)
x(:,:) = realnoiselet(x(:,:));