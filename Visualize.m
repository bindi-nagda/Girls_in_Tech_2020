% Visualizing bio-scan after solving the PDE
close all;
clear variables;
opts.DataRange = [1 1 4096 1];
z = readmatrix('Sol_1600.xlsx','Range',[1 1 4096 1]);
znew = zeros(66,66);
znew(2:65,2:65) =(reshape(z,64,64));
 
% Create 2D Grid
h = 1/(65);
  for i = 1:1:66
      x(i) = (i-1)*h;
      y(i) = (i-1)*h;
  end
[xx,yy] = meshgrid(x,y);
surf(xx,yy,znew); hold on;                % Plot Numerical solution
xlabel('x'); ylabel('y'); zlabel('z');
title('Patient''s bioscan showing tumor');

ylim([0,0.25]);
xlim([0,0.5]);
