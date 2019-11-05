% Plot the cumulative reward and proportion of users as 3D function of
% policy (p,q) . P1P2optimize.m is the file which optimizes one of the
% plots subject to some constraint on the other.

b00 = 0.7;
b01 = 0.1;
b10 = 0.1;
b11 = 0.7;
z0 = 0.5;
aTotal = 50;
T = 500;
[X,Y] = meshgrid(0:0.1:1);
[Z1,Z2] = arrayfun(@(x,y) rewardAndProp(b00,b01,b10,b11,x,y,z0,T,aTotal),X,Y);
%figure(3),surf(X,Y,Z1)
figure(4),surf(X,Y,Z2)