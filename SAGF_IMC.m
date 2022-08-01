function [V] = SAGF_IMC(A_,label,options)
% Incomplete Multi-view Clustering with Sample-level Auto-weighted Graph Fusion.
% IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.
P=length(A_);
numClust=numel(unique(label)); 
numiter = 10; 
G=options.G; 
n = size(G{1},1);
lambda = options.lambda;
S = zeros(n,n); B=cell(1,P); A=cell(1,P);
for p =1:P
    A{p}=G{p}*A_{p}*G{p}';
    S = S + A{p};
    tem = ones(1,n);
    tem(options.ind_0{p})=0;
    B{p}=diag(tem);
end
% S=S/num_views;
S = (S+S')/2;
DA = diag(sum(S));
L = DA - S;
[V, ~, ~] = eig1(L, numClust, 0);
k = 2;
 
while(k<=numiter)   
    fprintf('Running iteration %d\n',k-1);  
    % --- updata S ----
    ABW=0; BW=0; W=cell(1,P);
    for p = 1:P
        tem=S-A{p};
        W{p}=diag(0.5./sqrt(sum(tem.*tem,1)+eps));  %  set W
        ABW=ABW+A{p}*B{p}*W{p}; 
        BW=BW+B{p}*W{p};
    end       
    VV = L2_distance_1(V',V');  
    S=S.*(2*ABW)./max(2*S*BW+lambda*VV,eps);
    S=S*diag(1./sum(S,1)); 
    % --- updata V ----
    tem = diag( sum( (S + S.')/2 ) );
    L = tem - (S + S')/2 ;
    V = eig1(L, numClust, 0); 
    k = k+1;
end