% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function[Du] = ProxTVnorm(Du, gamma, mode, argin)
% Du1 = Du;
% 
% if ~exist('mode','var')
%     mode = 0;
% end
% 
% if mode == 1
%     insz = argin{1};
%     bsz = argin{2};
%     Du = reshape( Du , [ insz(1) , insz(2) * insz(3) ] );
%     Du = mat2cell( Du , bsz(1)*ones(1,insz(1)/bsz(1)) , bsz(2)*ones(1,insz(2)*insz(3)/bsz(2)) );
%     Du = cellfun( @(z) reshape( z , [1,1,numel(z)] ) , Du , 'UniformOutput', false);
%     Du = cell2mat(Du);
% end
% 
% n = size(Du);
% onemat = ones(n(1:2));
% thresh = ((sqrt(sum(Du.^2, 3))).^(-1))*gamma;
% thresh(thresh > 1) = 1;
% coef = (onemat - thresh);
% 
% for k = 1:n(3)
%     Du(:,:,k) = coef.*Du(:,:,k);
% end
%     
% if mode == 1
%    Du = mat2cell( Du , 1*ones(1,insz(1)/bsz(1)) , 1*ones(1,insz(2)*insz(3)/bsz(2)) , bsz(1)*bsz(2) );
%    Du = cellfun( @(z) reshape( z , bsz ), Du , 'UniformOutput', false);
%    Du = cell2mat(Du);
%    Du = reshape( Du , [ insz(1) * insz(2) , insz(3) ]  );
% end

if ~exist('mode','var')
    mode = 0;
end
n = size(Du);
if mode == 0
    onemat = ones(n(1:2));
    thresh = ((sqrt(sum(Du.^2, 3))).^(-1))*gamma;
    thresh(thresh > 1) = 1;
    coef = (onemat - thresh);
    
    for k = 1:n(3)
        Du(:,:,k) = coef.*Du(:,:,k);
    end
elseif mode == 1   
    insz = argin{1};
    bsz = argin{2};
    Du = reshape( Du , [ insz(1) , insz(2) * insz(3) ] );
    DuL2 = blockproc( Du , bsz , @(z) norm(z.data(:)) );
    
    onemat = ones(size(DuL2));
    thresh = (DuL2.^(-1))*gamma;
    thresh(thresh > 1) = 1;
    coef0 = (onemat - thresh);
    coef = kron(coef0,ones(bsz));
    
    Du = coef.*Du;
    Du = reshape( Du , [ insz(1) * insz(2) , insz(3) ]  );
end
