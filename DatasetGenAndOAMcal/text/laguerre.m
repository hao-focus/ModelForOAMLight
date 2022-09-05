function y=laguerre(n,l,x)

if prod(size(n))>1
    error('Laser Toolbox: HERMITE: N should be a non-negative scalar integer.');
end;

if (n<0 || abs(round(n)-n)>0)
    abs(round(n)-n)>0
   error('Laser Toolbox: HERMITE: N should be a non-negative integer.');
end;

if prod(size(l))>1
    error('Laser Toolbox: HERMITE: L should be a non-negative scalar integer.');
end;

if (l<0 || abs(round(l)-l)>0)
    abs(round(n)-n)>0
   error('Laser Toolbox: HERMITE: L should be a non-negative integer.');
end;

if n==0
   y=1;
elseif n==1,
   y=-1*x+1+l;
elseif n>1
   y= (2*n+l-1-x)./n.*laguerre(n-1,l,x) - (n+l-1)/n*laguerre(n-2,l,x);
end;

