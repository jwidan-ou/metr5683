# Library of useful functions for weather radar applications
# Author: David Bodine

#Aspect ratio relationship from Pruppacher and Beard (1970)
def aspect_ratio_pru70(dias):
	asp_ratio = 1.03 - 0.062*dias
	return asp_ratio

#Aspect ratio relationship from Beard and Chuang (1987)
def aspect_ratio_beard87(dias):
	asp_ratio = 1.0048 + 5.7e-4*dias - 2.628e-2*dias**2 + 3.682e-3*dias**3 - 1.677e-4*dias**4
	return asp_ratio

#Complex relative permittivity calculation for water
#Based on Cole and Cole 1941 and Ray 1972
#Inputs: lam (cm) and t_w (degrees C)
#Outputs: Complex dielectric factor (ep) and refractive index (m)
def complex_perm_water(lam,t_w):
	import numpy as np
	es=78.54*(1-4.579e-3*(t_w-25)+1.19e-5*(t_w-25)**2-2.8e-8*(t_w-25)**3);
	einf=5.27137+0.021647*t_w-0.00131198*t_w**2;
	alp=-16.8129/(t_w+273)+0.0609265;
	lam_s=0.00033836*np.exp(2513.98/(t_w+273));
	sigma=12.5664e8;

	#Calculate Real part of the dielectric factor for water
	ep_rw=einf+((es-einf)*(1+(lam_s/lam)**(1-alp)*np.sin(alp*np.pi/2)))/(1+2*(lam_s/lam)**(1-alp)*np.sin(alp*np.pi/2)+(lam_s/lam)**(2*(1-alp)));

	#Calculate Imaginary part of the dielectric factor for water
	ep_iw=((es-einf)*(lam_s/lam)**(1-alp)*np.cos(alp*np.pi/2))/(1+2*(lam_s/lam)**(1-alp)*np.sin(alp*np.pi/2)+(lam_s/lam)**(2*(1-alp)))+sigma*lam/(18.8496*10**10);
    
	#Calculate refractive index (real and imaginary components nr and ni)
	nr_water=(1/2*ep_rw + 1/2*(ep_iw**2 + ep_rw**2)**(1/2))**(1/2);
	ni_water=(2*(ep_rw/2 + (ep_iw**2 + ep_rw**2)**(1/2)/2)**(3/2) - 2*ep_rw*(ep_rw/2 + (ep_iw**2 + ep_rw**2)**(1/2)/2)**(1/2))/ep_iw;

	ep = ep_rw - ep_iw*1j
	m = nr_water - ni_water*1j
	return ep, m

#Same as the function above except for ice
def complex_perm_ice(lam,t_ice):
	import numpy as np
	es=203.168+2.5*t_ice+0.15*t_ice**2
	einf=3.168
	alp=0.288+0.0052*t_ice+0.00023*t_ice**2
	lam_s=9.990288e-4*np.exp(13200/((t_ice+273)*1.9869))
	sigma=1.26*np.exp(-12500/((t_ice+273)*1.9869))

	#Calculate Real part of the dielectric factor for ice	
	ep_ri=einf+((es-einf)*(1+(lam_s/lam)**(1-alp)*np.sin(alp*np.pi/2)))/(1+2*(lam_s/lam)**(1-alp)*np.sin(alp*np.pi/2)+(lam_s/lam)**(2*(1-alp)))

	#Calculate Imaginary part of the dielectric factor for ice
	ep_ii=((es-einf)*(lam_s/lam)**(1-alp)*np.cos(alp*np.pi/2))/(1+2*(lam_s/lam)*(1-alp)*np.sin(alp*np.pi/2)+(lam_s/lam)**(2*(1-alp)))+sigma*lam/(18.8496*10**10)


	#Calculate refractive index (real and imaginary components nr and ni)
	nr_ice=(1/2*ep_ri + 1/2*(ep_ii**2 + ep_ri**2)**(1/2))**(1/2)
	ni_ice=(2*(ep_ri/2 + (ep_ii**2 + ep_ri**2)**(1/2)/2)**(3/2) - 2*ep_ri*(ep_ri/2 + (ep_ii**2 + ep_ri**2)**(1/2)/2)**(1/2))/ep_ii

	m = nr_ice - 1j*ni_ice
	ep = ep_ri - 1j*ep_ii
	return ep, m

#Calculates shape factor for an oblate spheroid
#Aspect ratio assumed to be of the form b/a (i.e., < 1 for oblate spheroid)
def shape_factor_oblate(asp_ratio):
	import numpy as np
	gamma = 1/asp_ratio #(a/b)
	f = np.sqrt(gamma**2 - 1)
	la = (1+f**2)/f**2*(1-(np.arctan(f)/f))
	lb = (1-la)/2
	return la, lb

#Calculates scattering amplitudes for major and minor axis (sbb,saa) where
#b is the major axis and a is the minor axis
#Inputs include diameter (dia, mm), wavelength (lam, mm), shape factor,
#and complex relative permittivity
def scat_amp(dias,lam,la,lb,ep):
	import numpy as np
	sbb = np.pi**2*dias**3/(6*lam**2)*(1/(lb+1/(ep - 1)))
	saa = np.pi**2*dias**3/(6*lam**2)*(1/(la+1/(ep - 1)))
	return saa, sbb

#This function calculates the radar beam height
#Input range is assumed to be in km
#Input elevation angle is assumed to be in degrees
def bh_calc(r_km,ele):
        import numpy as np
        a=6.371e3 #Earth's Radius
        ae=4/3*a #Effective Earth's radius
        bh=np.sqrt(r_km**2+ae**2+2*r_km*ae*np.sin(ele*np.pi/180))-ae;
        return bh

