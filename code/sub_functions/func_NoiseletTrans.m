% Only for 2^n square image

function[x] = func_NoiseletTrans(x)
for i=1:size(x,3)
    x(:,:,i) = realnoiselet(x(:,:,i));
end
x = x/size(x,1);