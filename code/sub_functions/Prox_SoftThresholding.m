function y = Prox_SoftThresholding(x,T,v)

y = v + sign(x-v).*max(abs(x-v)-T,0);
