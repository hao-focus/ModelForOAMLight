close all
clear 
clc
%% initializing global parameters
tic
format long
lambda=1;
k=2*pi/lambda;
w= 7.113; 
rho_max=106.7 / 2;
mesh=200;
z=0;max_oam=10;
%% meshgrid
x1=linspace(-rho_max,rho_max,mesh);
y1=linspace(-rho_max,rho_max,mesh);
[X1,Y1]=meshgrid(x1,y1);
%% size of dataset
M=50;
Ntr=500;
Nva=100;
Nte=100;
%% storage path
filepath1='C:\DataSim\train\';
filepath2='C:\DataSim\val\';
filepath3='C:\DataSim\test\';
%% generating dataset
LG_general = zeros(21,mesh,mesh);
for jj = -10 :1: 10
    LG_temp = LGbeamV2(0,jj,X1,Y1,w,k,z,0);
    LG_general(jj+11,:,:) = LG_temp;
end
parfor ntr=1:Ntr
    LG = zeros(mesh,mesh);
    labels=abs(normrnd(0,0.2,1,2*max_oam+1));
    label_sum=sum(labels);
    labels=labels/label_sum;
    for i = 1:M
        E1=zeros(mesh,mesh);
        for index = -10 : 1 : 10
            LG(:,:) = LG_general(index+11,:,:);
            E1 = E1 + sqrt(labels(index+11))*exp(1i*rand(1)*pi)*LG;
        end
        E1 = E1 / max(max(abs(E1)));
        filename=sprintf('_label_x=%d_index=%d',ntr,i);
        name1=strcat(filepath1,'train',filename,'.mat');
        parsave(name1,E1,labels);
    end
end
parfor nva=1:Nva
    E1=zeros(mesh);
    labels=abs(normrnd(0,0.2,1,2*max_oam+1));
    label_sum=sum(labels);
    labels=labels/label_sum;
    for i = 1:M
        E1=zeros(mesh,mesh);
        for index = -10 : 1 : 10
            LG = LG_general(index+11,:,:);
            LG = reshape(LG,[mesh,mesh]);
            E1 = E1 + sqrt(labels(index+11))* exp(1i*rand(1)*pi)*LG;
        end
        E1 = E1 / max(max(abs(E1)));
        filename=sprintf('_label_x=%d_index=%d',nva,i);
        name1=strcat(filepath2,'val',filename,'.mat');
        parsave(name1,E1,labels);
    end
end
parfor nte=1:Nte
    E1=zeros(mesh);
    labels=abs(normrnd(0,0.2,1,2*max_oam+1));   
    label_sum=sum(labels);
    labels=labels/label_sum;
    for i = 1:M
        E1=zeros(mesh,mesh);
        for index = -10 : 1 : 10
            LG = LG_general(index+11,:,:);
            LG = reshape(LG,[mesh,mesh]);
            E1 = E1 + sqrt(labels(index+11))*exp(1i*rand(1)*pi)*LG;
        end
        E1 = E1 / max(max(abs(E1)));
        filename=sprintf('_label_x=%d_index=%d',nte,i);
        name1=strcat(filepath3,'test',filename,'.mat');
        parsave(name1,E1,labels);
    end
end
toc
%% 
function[]=parsave(dir,E,OAM_s)
save(dir,'E','OAM_s','-v6');
end

