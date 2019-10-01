%% Find policy s.t cumulative reward is maximized subject to constraint
% This code finds the optimal value of (p,q) policy s.t reward at a deadline is maximized 
% subject to the constraint that the proportion of users of preference 0 is greater than 
% a certain threshold zThresh. 
% Plots the obtained optimal policy vs values of zThresh.

b00 = 0.8;
b01 = 0.1;
b10 = 0.5;
b11 = 0.7;
z0 = 0.5;
zThresh = 0:0.01:0.99;
aTotal = 20;
T = 2000000;
policyArr1 = [];
policyArr2 = [];

for i = zThresh

    pol = optimvar('pol',1,2,'LowerBound',0,'UpperBound',1);

    [obj,cons] = rewardAndProp(b00,b01,b10,b11,pol(1),pol(2),z0,T,aTotal);

    obj = -obj;

    prob = optimproblem('Objective',obj);

    nlcons = cons >= i;

    prob.Constraints.circlecons = nlcons;

    %showproblem(prob);

    x0.pol = [0.5 0.5];
    [sol,fval,exitflag,output] = solve(prob,x0);

    policyArr1 = [policyArr1,sol.pol(1)];
    policyArr2 = [policyArr2,sol.pol(2)];
end
figure(1)
plot(zThresh,policyArr1,zThresh,policyArr2)

