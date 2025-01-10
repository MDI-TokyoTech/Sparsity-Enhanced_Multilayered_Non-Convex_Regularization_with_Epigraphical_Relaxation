% Promote sparsity by setting the micro value to 0
% (not used in paper)
function zstar = mask_referenceData(zstar, epsilon)
zstar(zstar<=epsilon) = 0;