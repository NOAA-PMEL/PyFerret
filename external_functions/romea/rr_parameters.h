*           rr_parameters.h
*
*           Rick Romea
*           Jan. 5 2000
*
*     Parameters for the RR external functions
*
*******************************************************************

      ! Something good to eat!

      REAL       RR_pi
      PARAMETER (RR_pi = 3.1415927)
      
      ! Earth's radius (cm)

      REAL       RR_radius
      PARAMETER (RR_radius = 6370.e5) 
      
      ! Convert: degrees --> radians

      REAL       Degrees_to_radians
      PARAMETER (Degrees_to_radians = RR_pi/180.)

      ! Coriolis term  (1/sec)

      REAL       Two_Omega
      PARAMETER (Two_Omega = 2. * RR_pi / 43082.)  
	  
      ! Convert: deg --> cm   

      REAL       Longitude_to_cm
      PARAMETER (Longitude_to_cm = RR_pi*RR_Radius/180.)

      ! Convert: deg --> cm   
     
      REAL       Latitude_to_cm
      PARAMETER (Latitude_to_cm = RR_pi*RR_Radius/180.)
      
      ! Convert: m --> cm

      REAL       Meters_to_cm
      PARAMETER (Meters_to_cm = 100.)

      ! Density (g/cm^3)

      REAL       Rho_zero
      PARAMETER (Rho_zero = 1.035) 
      
      ! Time conversion :  Convert: month --> sec
	
      ! NOTE: I have set  sec_per_month = 1, so the functions return 
      !       eg.,  cm/s/s, not cm/s/month
  
      REAL       sec_per_month
      PARAMETER (sec_per_month = 1.)
      ! PARAMETER (sec_per_month = 60.*60.*24.*30.)
							   
      ! ah = constant lateral diffusion coeff for tracers (cm**2/sec)

      REAL       ah 
      PARAMETER (ah = 2.e7)  
	   
      ! am = constant lateral viscosity coeff for momentum ( cm**2/sec)
		 
      REAL       am 
      PARAMETER (am = 1.e7)  

      ! fricmx = maximum  diffusion coefficient (cm**2/s) 

      REAL       fricmx      
      PARAMETER (fricmx = 50.)

      ! diff_cbt_limit = large diffusion coefficient (cm**2/sec) 
																         
      REAL       diff_cbt_limit
      PARAMETER (diff_cbt_limit = fricmx)
      
      ! diff_cbt_back  = background diffusion coefficient (cm**2/s)
	  
      REAL       diff_cbt_back 
      PARAMETER (diff_cbt_back = 0.1)
    	  
      !  visc_cbu_limit = largest viscosity (cm**2/sec)
      
      REAL       visc_cbu_limit
      PARAMETER (visc_cbu_limit = fricmx)
      
      ! visc_cbu_back  = background viscosity (cm**2/s)  
	  
      REAL       visc_cbu_back 
      PARAMETER (visc_cbu_back = 1.)

      ! wndmix = min value for diffusion coefficient at surface to 
      !          simulate high freq wind mixing. (cm**2/sec)  
	  										  
      REAL       wndmix        
      PARAMETER (wndmix = 10.)

      ! gravity (cm/sec**2)
     
      REAL       gravity
      PARAMETER (gravity = 980.6) 

      REAL       grav
      PARAMETER (grav = gravity) 

      ! small value
	
      REAL       epsln
      PARAMETER (epsln = 1.e-25)

	  
      ! Temporary variable

      REAL	 RR_temp1
      PARAMETER (RR_temp1 = -Meters_to_cm*gravity/2./Rho_zero)

      ! 
      REAL       dyt_ref  ! dyt(50)
      PARAMETER (dyt_ref = .333*Latitude_to_cm)


