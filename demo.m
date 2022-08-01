% Incomplete Multi-view Clustering with Sample-level Auto-weighted Graph Fusion.
% IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022.
clear all;
%% =====================  setting =====================
options.dataStr='3-sources';
load(char(options.dataStr));
options.lambda=10; % controls the learned graph, needed to be set
options.miss=0.1;  % missing rate 
%% =====================  run =====================
for fold_i=1:30  
    options.fold_i=fold_i;
    [X_part,options]=gen_incom_data(data,options); 
	%% =====  Data normalization ======    
    X_part=data_normalization(X_part);
    %% predefines the view-specific graph 
    kk=20;  islocal_1=1;   
    A_=cell(1,length(data));
    for i=1:length(data)
    A_{i}=Updata_Sv(X_part{i},numel(unique(label)),kk,islocal_1);
    end
    %% main function
    V=SAGF_IMC(A_,label,options);
	%% =====================  result =====================
    [ac1, nmi1, Pri1]=print_results(V, label', numel(unique(label)), 1); 
end

function [data,parameter]=gen_incom_data(data,parameter)
numOfDatas=size(data{1},2);
data0=data; 
nv=length(data);
 
percentDel=parameter.miss;
Datafold = [char(parameter.dataStr),'_miss_',num2str(percentDel),'.mat']; 
load(Datafold);  
ind_folds = folds{parameter.fold_i};
X_0=data0;
clear data;
WI = eye(numOfDatas); 
for iv = 1:nv  
    ind_0{iv} = find(ind_folds(:,iv) == 0); 
    ind_1{iv} = find(ind_folds(:,iv) == 1); 
    data{iv} = data0{iv}(:,ind_1{iv});   
    X_0{iv}(:,ind_0{iv}) = 0; 
    W1{iv} = WI(:,ind_1{iv});   
    W0{iv} = WI(:,ind_0{iv});  
end
parameter.ind_0=ind_0;
parameter.G=W1;
% parameter.W0=W0;
end

function data=data_normalization(data)
for v = 1:length(data)
    data{v} = data{v}*diag(sparse(1./sqrt(sum(data{v}.^2))));
end  
end


