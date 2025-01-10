function [DuL2] = func_L2_for_DVTV(Du, wlumi)
[v, h, ~, ~] = size(Du);
DuL2 = zeros([v, h, 2]);

for i = 1:v
    for j = 1:h
        DuL2(i,j,1) = wlumi*sqrt(sum(Du(i,j,1,:).^2, 4));
        DuL2(i,j,2) = sqrt( sum(Du(i,j,2,:).^2, 4)  +  sum(Du(i,j,3,:).^2, 4) );
    end
end