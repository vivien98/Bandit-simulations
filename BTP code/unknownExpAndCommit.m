b00 = 0.9;
b01 = 0.3;
b10 = 0.3;
b11 = 0.9;
z0 = 0.5;
A = 20;
delta0 = abs(1-b00-b01);
delta1 = abs(1-b11-b10);

T = 500;
thresh = 1:(T-1);
opinion = zeros(1,T-1);
optimalOpinion = zeros(1,T-1);
m = zeros(1,T-1);
j = T-1;
% for j = 1:T-1
%     j
    for i = 1:(j)
       i
       [dump,temp] = rewardAndProp(b00,b01,b10,b11,0.5,0.5,z0,i,A);
       p = 1;
       q = 0;
       [dump,zRight] = rewardAndProp(b00,b01,b10,b11,p,q,temp,T-i,A+i);
       [dump,zWrong] = rewardAndProp(b00,b01,b10,b11,1-p,1-q,temp,T-i,A+i);
       opinion(i) = zRight*(1-exp(-i*delta0*delta0)) + zWrong*exp(-i*delta0*delta0); %% constant factor in the hoeffding exp may be wrong
       %optimalOpinion(i) = rewardAndPropVoter(b00,b01,b10,b11,p,q,temp,T,A);
    end
%     [prop_opt,m(j)] = max(opinion);
% end
plot(thresh,opinion);
%plot(thresh,m);

