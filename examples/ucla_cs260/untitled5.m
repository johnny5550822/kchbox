plot(systo(1,:))
[A,D] = dwt(systo(1,:),'sym4');
figure
plot(A)