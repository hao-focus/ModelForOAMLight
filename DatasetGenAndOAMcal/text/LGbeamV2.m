function [LGpl] = LGbeamV2(p,l,X,Y,w,k,z,phi0)
[phi,r] = cart2pol(X,Y);
f=w^2*k/2; 
R=z+f^2./z;  
W=w*(1+(z/f).^2).^0.5; 
C=(2*factorial(p)/(pi*factorial(p+abs(l))))^0.5;
Gouy=1i*k*(z+r.^2/2./R)-1i*(2*p+abs(l)+1)*atan(z/f);
LGpl=C./W.*exp(1i*l*phi).*(2^0.5./W.*r).^abs(l).*laguerre(p,abs(l),2./W.^2.*r.^2).*exp(-r.^2./W.^2).*exp(Gouy).*exp(1i*phi0);
M = max(max(abs(LGpl)));
LGpl = LGpl ./ M;
end