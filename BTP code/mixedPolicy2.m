function reward = mixedPolicy2(b00,b01,b10,b11,z0,T,Ts,aTotal)
[o1,o2]=mixedPolicy(b00,b01,b10,b11,z0,T,Ts,aTotal);
reward = o1;
