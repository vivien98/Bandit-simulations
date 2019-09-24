%% Find policy s.t cumulative reward is maximized subject to constraint
% This code finds the optimal value of (p,q) policy s.t reward at a deadline is maximized 
% subject to the constraint that the proportion of users of preference 0 is greater than 
% a certain threshold zThresh.

b00 = 0.8;
b01 = 0.1;
b10 = 0.5;
b11 = 0.7;
z0 = 0.5;
zThresh = 0.65;
aTotal = 20;
T = 2000000;
pol = optimvar('pol',1,2,'LowerBound',0,'UpperBound',1);

[obj,cons] = rewardAndProp(b00,b01,b10,b11,pol(1),pol(2),z0,T,aTotal);

obj = -obj;

prob = optimproblem('Objective',obj);

nlcons = cons >= zThresh;

prob.Constraints.circlecons = nlcons;

showproblem(prob);

x0.pol = [0.5 0.5];
[sol,fval,exitflag,output] = solve(prob,x0)

sol.pol