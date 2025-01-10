function xprj = Proj_Halfspace(x, eta, w, dir)
% xleft <= mu -> x - mu <= 0 -> [ 1, ... , 1 , -1 ] * [ x ; mu ] <=0
xprj = x;
ind = find( sum(w.*x,dir) > eta );
x0 = x(ind,:);
w0 = w(ind,:);
x0prj = project_hyperplane(x0, eta, w0, dir);
xprj(ind,:) = x0prj;
% x = [ xleft ; mu ];
% a = [ ones(length(xleft),1) ; -1 ];
% if sum(x(:)) > 0
%     x = x - sum(a.*x)/sum(a.*a)*a;
% end
% 
% xprj = x(1:end-1);
% muprj = x(end);