b = zeros(2);
b(1,1) = 0.7;
b(1,2) = 0.3;
b(2,1) = 0.4;
b(2,2) = 0.5;
p = 1;
q = 1;
z0 = 0.5;
N0 = 20;
T = 500;

d1 = p*(1-b(1,1)) + (1-p)*b(1,2);
d2 = q*(1-b(2,2)) + (1-q)*b(2,1);

time = 1:T;

nSim = 1;
%________________________________________________________________%

odeSol = @(t)(d2/(d1+d2) + (z0-d2/(d1+d2))*(1+t/N0).^(-d1-d2));

ode = odeSol(time);

%_________________________________________________________________%
outSim = zeros(2,T);
cnt = 1;
for j = [10,100]

Z0 = N0*z0;
sim = zeros(1,T);
samp1 = rand(j,T);
samp2 = rand(j,T);
samp3 = rand(j,T);

for n = 1:j
    zList = zeros(1,T);
    ZList = zeros(1,T);
    tot = N0;
    prev = z0;
    prevZ = Z0;
    for i = 1:T
        type = (samp1(n,i) > prev);
        if type == 0
            arm = (samp2(n,i) > p);
        else
            arm = (samp2(n,i) < q);
        end
        
        rew = (samp3(n,i) < b(type+1,arm+1));
        tot = tot + 1;
        
        if arm == 0
            ZList(i) = prevZ + rew;    
        else
            ZList(i) = prevZ + 1-rew;            
        end
        zList(i) = ZList(i)/tot;
        prevZ = ZList(i);
        prev = zList(i);
    end
    sim = sim + zList;
     
end
outSim(cnt,:) = sim/j;
cnt = cnt + 1;
end

%_________________________________________________________________%

plot(time,ode,time,outSim(1,:),time,outSim(2,:));
