% plot the cumulative reward and proportion of users vs T0 for different
% values of T0
b00 = 0.8;
b01 = 0.1;
b10 = 0.5;
b11 = 0.7;
z0 = 0.5;
aTotal = 20;
T = 2000000;
Ts = 1:1000:200000;
%% IMPORTANT : ensure that rewardAndProp function does not have the zt line commented !!
[r,pr]=arrayfun(@(x) mixedPolicy(b00,b01,b10,b11,z0,T,x,aTotal),Ts); 

figure(1),plot(Ts,r);
figure(2),plot(Ts,pr);
