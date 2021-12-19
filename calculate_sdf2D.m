function [DTgrid,GXgrid,GYgrid,NVgrid, XX,YY, DT,NV_vector,NV_theta,IDgrid, FX,FY] = sdf(Step, observation, normalvector)
%dualTSDF: generate 2D dual TSDF according to the observation and the normal
%vector
%    input: observation: 2*N
%           normalvector: 2*N
%
%    output: 



% threshold
dist_threshold = 500;
theta_threshold = pi/3;
theta_threshold_normalProduct = cos(theta_threshold);

% grid initialization
% Step = 1;
Y = 1:Step:500;
X = 1:Step:500;
nX = length(X);
nY = length(Y);

% load model contour and normal
verX = observation(1,:);
verY = observation(2,:);
NNT = normalvector;

% store distance and X Y position
DT = dist_threshold*ones(nX,nY); 
XX = zeros(nX,nY);
YY = zeros*ones(nX,nY);


% store corresponding observation index and normal vector (theta)
NV_vector = cell(nX,nY);
NV_theta = zeros(nX,nY);
ID = -10000*ones(nX,nY);


for j=1:nY
    for i=1:nX        
        % store the position of this grid (id)
        ix = (i-1)*Step+X(1);   % grid
        jy = (j-1)*Step+Y(1);
        XX(i,j) = ix;
        YY(i,j) = jy;
        
        Map(1,:) = verX-ix;     % for calculating distance
        Map(2,:) = verY-jy;
        Map = Map.^2;           % distance
        dist = sqrt(sum(Map));
        [a1,b1] = min(dist);       % min distance
 
        
        N1_1 = [ix-verX(b1),jy-verY(b1)];    % the vector between this grid and the point on the surface with min distance
        N1_1norm = N1_1/norm(N1_1);          % get the normal vector
        N2_1 = NNT(:,b1);                    % the normal of this point for getting sign
        
        innerProduct_normalvector = N1_1norm*N2_1;
        if(innerProduct_normalvector>1e-6)
            S1 = -1;
        elseif (innerProduct_normalvector<-1e-6)
            S1 = 1;
        else
            S1 = 0;
        end

        % flag to add constraint to calculate SDF
        flag_calculateSDF1 = true;
        % distance constraint
        if(a1>dist_threshold)
%             DT(i,j) = dist_threshold;
            flag_calculateSDF1 = false;    
        end
        % normal vector theta
%         if 0
%         innerProduct_normalvector = N1_1norm*N2_1;   % normal vector with the length of 1, no need to normalize
%         if(abs(innerProduct_normalvector)<theta_threshold_normalProduct)
%             flag_calculateSDF1 = false; 
%         end
%         end

        % for smooth pixel: 去掉之后会使得distance field不平滑
        if(a1 < 3)
            flag_calculateSDF1 = true; 
        end
        
%         flag_calculateSDF1 = true;
        % store data
        if(flag_calculateSDF1)
            DT(i,j) = a1*S1;     % signed distance. We can see that the normal is only for calculating sign, not used for calculating point2plane distance.
            NV_vector{i,j} = N2_1;       % store theta
        else
            DT(i,j) = dist_threshold*S1;   % add sign
        end
        

        % store N2_theta
        NV_theta(i,j) = atan2(N2_1(2), N2_1(1));
        ID(i,j) = b1;
    end
end

%% calculate gradient
[FY,FX] = gradient(DT,Step,Step);


DTgrid = griddedInterpolant(XX,YY,DT);
GXgrid = griddedInterpolant(XX,YY,FX);
GYgrid = griddedInterpolant(XX,YY,FY);
NVgrid = griddedInterpolant(XX,YY,NV_theta);

IDgrid = griddedInterpolant(XX,YY,ID, 'nearest');

end
