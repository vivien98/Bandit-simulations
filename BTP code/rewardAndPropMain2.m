b00 = 0.8;
b01 = 0.1;
b10 = 0.5;
b11 = 0.7;
z0 = 0.5;
aTotal = 20;
T = 2000000;
[X,Y] = meshgrid(0:0.1:1);
[Z1,Z2] = arrayfun(@(x,y) rewardAndProp(b00,b01,b10,b11,x,y,z0,T,aTotal),X,Y);
figure(1),surf(X,Y,Z1)
figure(2),surf(X,Y,Z2)