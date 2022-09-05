function [OAM_s] = exp_spec(E, max_oam,mesh,rho_max)
% E is normalized
x1=linspace(-rho_max,rho_max,mesh);
y1=linspace(-rho_max,rho_max,mesh);
[X1,Y1]=meshgrid(x1,y1);
E_r=sqrt(X1.^2 + Y1.^2);
E_phi=angle(X1 + 1i*Y1) + pi;
spectrum1=zeros(2*max_oam+1,1);
for l=-max_oam:1:max_oam
    E_xy = E.*exp(-1i*l*E_phi);
    r_inter=zeros(200,1);
    r_final=zeros(200,1);
    for kk = 1 : 200
        rmin = 0.01*rho_max + (kk-1) * 4.95e-3 * rho_max;
        rmax = 0.01*rho_max + (kk ) * 4.95e-3 * rho_max; %0.99rho_max/200
        r_final(kk) = (rmin+rmax)/2;
        iter=1;
        for xx = 1 : mesh
            for yy = 1 : mesh
                if E_r(xx, yy) >= rmin && E_r(xx, yy) < rmax
                    E_temp(iter)=E_xy(xx,yy);
                    phi_temp(iter)=E_phi(xx, yy);
                    iter = iter + 1;
                end
            end
        end
        if exist('phi_temp') 
            [Phi_final, sorting]=sort(phi_temp);
            E_final = E_temp(sorting);
            ik=1;
            if length(phi_temp)>200
                alpha=1;
                ak = ceil(length(phi_temp)/200);
                for temp2 = 1:length(phi_temp)
                    if ik == ak
                        ik=1;
                        if ak*alpha <= length(phi_temp)
                            temp_phi(alpha) = mean(phik);
                            temp_E(alpha) = mean(Ek);
                            clear Ek phik
                            alpha= alpha + 1;
                        else
                            break
                        end
                    else
                        Ek(ik)=E_final((alpha-1)*ak+ik);
                        phik(ik)=Phi_final((alpha-1)*ak+ik);
                        ik = ik + 1;
                    end
                end
                phi_inter=1/sqrt(2*pi)*trapz(temp_phi, temp_E);
                r_inter(kk)=phi_inter;
            else 
                phi_inter=1/sqrt(2*pi)*trapz(Phi_final,E_final);
                r_inter(kk)=phi_inter;
            end
            clear E_temp phi_temp temp_phi temp_E Ek phik
        end
    end
    r_interf =abs(r_inter).^2.*r_final;
    E_f = trapz(r_final, r_interf);
    spectrum1(l+max_oam+1)=E_f;
end
k = sum(spectrum1);
spectrum1 = spectrum1/k;
OAM_s = spectrum1;
end