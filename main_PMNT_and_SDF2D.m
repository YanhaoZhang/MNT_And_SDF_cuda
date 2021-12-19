% close all
clear

load('aorta_silhouette.mat')


%% traditional method
% get 2D contour using MNT
Boundary = [];
Normal = [];
[B,~] = bwboundaries(aorta_silhouette,'holes');
num_B = length(B);
boundary2d_tmp1 = cell(1,num_B);
normal_tmp1 = cell(1,num_B);
if_flip = 1;        % unify the boundary order
for i=1:num_B
    boundary2d_tmp2 = B{i};
    observation = [boundary2d_tmp2(:,1), boundary2d_tmp2(:,2)];
    
    
    
    NormalVectorLines = 10;
    Lines_obser=[(1:(size(observation,1)-NormalVectorLines))', (NormalVectorLines+1:size(observation,1))'];
    normal_observation = LineNormals2D(observation,Lines_obser);   % calculate normal vector
    Boundary = [Boundary; observation];
    Normal = [Normal; normal_observation];
end


% calculate SDF
step = 1;
[~,~,~,~, XX,YY,DT,~, ~,~, ~,~] = calculate_sdf2D(step, Boundary', Normal');




%% PMNT and SDF using cuda
[cuSDF, ~, ~, ~, ~, ~, cuBoundx, cuBoundy] = main_sdf(aorta_silhouette);

% rewrite data to get boundary pixel
cuBoundx(cuBoundx==0) = [];
cuBoundy(cuBoundy==0) = [];
cuBound = [cuBoundx, cuBoundy];


%% check SDF result
figure
subplot(1,3,1)
imshow(aorta_silhouette); hold on
plot(Boundary(:,2),Boundary(:,1),'r.','Markersize',3); 
plot(cuBound(:,2),cuBound(:,1),'b.','Markersize',3); 
legend('MNT', 'PMNT')
axis equal
title('Silhouette and Boundary Pixels')

subplot(1,3,2)
contour(XX,YY,cuSDF);
axis equal
title('SDF by CUDA')  

subplot(1,3,3)
contour(XX,YY,DT);
axis equal
title('SDF by Matlab')  






    
    
  