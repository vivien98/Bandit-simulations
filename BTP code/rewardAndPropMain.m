b00 = 0.8;
b01 = 0.1;
b10 = 0.5;
b11 = 0.7;
z0 = 0.5;
aTotal = 20;
T = 2000000;
Ts = 1:10000:2000000;
% p=1;
% q=1;
% [rg,prg] = rewardAndProp(b00,b01,b10,b11,p,q,z0,T,A);
% p=(b00 > 1-b01);
% q=(b11 < 1-b10);
% [ro,pro] = rewardAndProp(b00,b01,b10,b11,p,q,z0,T,A);
[r,pr]=arrayfun(@(x) mixedPolicy(b00,b01,b10,b11,z0,T,x,aTotal),Ts);
figure(1),plot(Ts,r);
figure(2),plot(Ts,pr);

% rew = [ro,r,rg];
% prop = [pro,pr,prg];
% 
% rew
% prop
