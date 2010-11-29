******************************************************************
*
*           RR_cross_spectrum_abs.F
*
*             Rick Romea   Oct.8, 2000
*
*
* Cross-spectral program
*
* INPUT:
* arg_1  X1 (I, J, L=1:N) Real time series 1 		
* arg_2  X2 (I, J, L=1:N) Real time series 2	
* arg_3  Nf   Number of frequencies per band 	 
* arg_4  P    Percentage level for significance (90, 95 or 99)
*
*        handed along:  Number of data points N
*       		Sampling interval of data points
*
* OUTPUT:  ANSWER (I,J,K=1:11,L=1:Nb)
*
* ANSWER (I,J,K=1,L=Nb)	 Power in  band1	         
* ANSWER (I,J,K=2,L=Nb)	 Power in band2	                 
* ANSWER (I,J,K=3,L=Nb)	 Series 1: Upper confidence level	
* ANSWER (I,J,K=4,L=Nb)  Series 1: Lower confidence level	
* ANSWER (I,J,K=5,L=Nb)	 Series 2: Upper confidence level	
* ANSWER (I,J,K=6,L=Nb)	 Series 2: Lower confidence level	
* ANSWER (I,J,K=7,L=Nb)  Coherence-squared in each band	
* ANSWER (I,J,K=8,L=Nb)  Phase in each band 	
* ANSWER (I,J,K=9,L=Nb) Significant coherence squared
* ANSWER (I,J,K=10,L=Nb) Upper confidence level for phase
* ANSWER (I,J,K=11,L=Nb) Lower confidence level for phase
*
*       handed along: 	Number of bands	 Nb
*                       frequency axis
*
******************************************************************

      SUBROUTINE RR_cross_spectrum_abs_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      PRINT*,'START INIT'

      CALL ef_set_desc(id,'Computes auto and cross-spectrum')
      CALL ef_set_num_args(id,4)
 
      ! x and y unchanged
      ! on output:  z-axis is ABSTRACT:  has k=1:12
      ! on output:  t-axis is CUSTOM: converted to frequency axis

      CALL ef_set_axis_inheritance(id,
     .   IMPLIED_BY_ARGS,IMPLIED_BY_ARGS,ABSTRACT, ABSTRACT)   
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      !  Set a work array

      CALL ef_set_num_work_arrays  (id, 8)

      ! Describe the input arguments

      ! arg_1  X1 (I, J, L=1:N) Real time series 1 		

      CALL ef_set_arg_name         (id, ARG1, 'S1')
      CALL ef_set_arg_desc         (id, ARG1, 'Series 1')
      CALL ef_set_arg_unit         (id, ARG1, ' ')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES,YES,NO,NO)

      ! arg_2  X2 (I, J, L=1:N) Real time series 2	

      CALL ef_set_arg_name         (id, ARG2, 'S2')
      CALL ef_set_arg_desc         (id, ARG2, 'Series 2')
      CALL ef_set_arg_unit         (id, ARG2, ' ')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, YES,YES,NO,NO)

      ! arg_3  Nf Number of frequencies per band 	 
      !  scalar
      
      CALL ef_set_arg_name         (id, ARG3, 'Nf')
      CALL ef_set_arg_desc         (id, ARG3, 
     .                  'Number of frequencies per band')
      CALL ef_set_arg_unit         (id, ARG3, ' ')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3, NO,NO,NO,NO)

      ! arg_4  P  Percentage level for significance (90, 95 or 99)
      !  scalar

      CALL ef_set_arg_name         (id, ARG4, 'P')
      CALL ef_set_arg_desc         (id, ARG4,
     .       'Percentage level for significance (90, 95 or 99)')
      CALL ef_set_arg_unit         (id, ARG4, ' ') 
      CALL ef_set_arg_type         (id, ARG4, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG4, NO,NO,NO,NO)

      PRINT*,'INIT DONE'
 
      END


C************************************************************************

      SUBROUTINE  RR_cross_spectrum_abs_work_size(id)
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INTEGER id
      INTEGER Npts
      INTEGER arg_lo_ss(4,EF_MAX_ARGS),arg_hi_ss(4,EF_MAX_ARGS),
     .                                 arg_incr(4,EF_MAX_ARGS)


      PRINT*,'START WORK_SIZE'
       
      !  Get the number of points in the time series = Npts

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr) 
      Npts = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1 

      PRINT*,'NPTS = ',Npts

      ! ef_set_work_array_dims(id,iarray,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi) 
      CALL ef_set_work_array_dims(id,1,1,1,1,1,1,1,1,4*Npts+1) 
      CALL ef_set_work_array_dims(id,2,1,1,1,1,1,1,1,Npts) 
      CALL ef_set_work_array_dims(id,3,1,1,1,1,1,1,1,Npts) 
      CALL ef_set_work_array_dims(id,4,1,1,1,1,1,1,1,Npts) 
      CALL ef_set_work_array_dims(id,5,1,1,1,1,1,1,1,Npts) 
      CALL ef_set_work_array_dims(id,6,1,1,1,1,1,1,1,Npts) 
      CALL ef_set_work_array_dims(id,7,1,1,1,1,1,1,1,Npts) 
      CALL ef_set_work_array_dims(id,8,1,1,1,1,1,1,1,2*Npts) 
      
      PRINT*,'END WORK_SIZE'
      END



C****************************************************************

      SUBROUTINE RR_cross_spectrum_abs_result_limits(id)
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INTEGER id

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), 
     .        arg_hi_ss(4,EF_MAX_ARGS),
     .         arg_incr(4,EF_MAX_ARGS)

      INTEGER Nb,Npts,Nf
      REAL    rNf


      PRINT*,'START RESULT_LIMITS'

      ! TIME AXIS

         !  Get the (integer) number of frequencies per band = Nf.

         CALL ef_get_one_val(id, ARG3, rNf)  
         Nf=NINT(rNf)

      PRINT*,'Nf = ',Nf

         !  Get the number of points in the time series = Npts.

         CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr) 

         Npts = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1 
     
      PRINT*,'NPTS = ',Npts


         ! Number of bands

         Nb = Npts/(2*Nf)

      PRINT*,'Nb = ',Nb

         CALL ef_set_axis_limits(id, T_AXIS, 1, Nb)

      ! ABSTRACT Z AXIS

         CALL ef_set_axis_limits(id, Z_AXIS, 1, 11)

      PRINT*,'END RESULT_LIMITS'

      END

C************************************************************

      SUBROUTINE RR_cross_spectrum_abs_compute(id,S1,S2,arg_3,arg_4,
     .                                ANSWER,WSAVE,a1,a2,b1,b2,X1,X2,C)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INTEGER id
      REAL S1  (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .          mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL S2  (mem2lox:mem2hix, mem2loy:mem1hiy, 
     .          mem2loz:mem2hiz, mem2lot:mem1hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .          mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4  (mem4lox:mem4hix, mem4loy:mem4hiy, 
     .          mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL ANSWER (memreslox:memreshix, memresloy:memreshiy, 
     .             memresloz:memreshiz, memreslot:memreshit)
      INTEGER  res_lo_ss (4)
      INTEGER  res_hi_ss (4)
      INTEGER  res_incr  (4)

      INTEGER  arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER  arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER  arg_incr  (4,EF_MAX_ARGS)

      INTEGER i,  j,  l
      INTEGER i1, j1, l1
      INTEGER i2, j2, l2

      REAL    WSAVE(wrk1lot:wrk1hit)
      REAL       a1(wrk2lot:wrk2hit)
      REAL       b1(wrk3lot:wrk3hit)
      REAL       a2(wrk4lot:wrk4hit)
      REAL       b2(wrk5lot:wrk5hit)
      REAL       X1(wrk6lot:wrk6hit)
      REAL       X2(wrk7lot:wrk7hit)
      COMPLEX     C(wrk8lot:wrk8hit)

      INTEGER Npts,Nf,Nb,kk
      REAL rNf,alpha,P,Coh2sig,lower,upper,PP,QQ,theta,GetStudentT  
      EXTERNAL GetStudentT

      PRINT*,'START COMPUTE'

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
         
      !  Get the number of points in the time series = Npts.

      Npts = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1 


      PRINT*,'NPTS = ',Npts

      !  Get the (integer) number of frequencies per band = Nf.

      CALL ef_get_one_val(id, ARG3, rNf)  
      Nf=NINT(rNf)
         
      PRINT*,'Nf = ',Nf

      ! Number of bands

      Nb = Npts/(2*Nf)


      PRINT*,'Nb = ',Nb

      ! Set up the cosine matrix

      PRINT*,'START CFFTI'

      CALL CFFTI(Npts,WSAVE)

      PRINT*,'END CFFTI'

      !  Significance level for coherence at P% with 2*Nf degrees of freedom
      !  Ref: R.O.R.Y.Thompson, Coherence Significance Levels, AMS, 1979

      CALL ef_get_one_val(id, ARG4, P)  
      alpha = 1.-P/100.
      Coh2sig= 1.-alpha**(1./FLOAT(Nf-1))


      !  Confidence intervals for the power spectral estimates
      !
      !   IF Gxx is the 'true' autospectral value, based on the 
      !    band-averaged estimate <Gxx>, 
      !    then, for Nf number of frequencies in band, 
      !    we can say with P certainty that
      !
      !  lower * <Gxx>  <=  Gxx  <=  upper * <Gxx> 
      !
      !       (Bendat & Piersol, p.286)  

      PRINT*,'START Confindence Intervals'
      
      CALL SpectralConfidenceIntervals(Nf,NINT(P),lower,upper)

      PRINT*,'lower = ',lower
      PRINT*,'upper = ',upper

      ! START

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
        j1 = arg_lo_ss(Y_AXIS,ARG1)
        j2 = arg_lo_ss(Y_AXIS,ARG2)
        DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           ! Copy the arrays
           
           l=1 
           DO l1=arg_lo_ss(T_AXIS,ARG1),arg_hi_ss(T_AXIS,ARG1)
              X1(l)=S1(i1,j1,1,l1)
              l=l+1
           ENDDO
            
           l=1 
           DO l2=arg_lo_ss(T_AXIS,ARG2),arg_hi_ss(T_AXIS,ARG2)
              X2(l)=S1(i2,j2,1,l2)
              l=l+1
           ENDDO

           ! Compute the raw spectral estimates

           PRINT*,'FFT - 1'

           CALL FFTreal(Npts,X1,a1,b1,WSAVE,C)

           PRINT*,'FFT - 2'

           CALL FFTreal(Npts,X2,a2,b2,WSAVE,C)

           ! Compute the auto-spectra, coherence-squared and the phase
          
           ! Band average


           PRINT*, 'START BAND AVERAGE'

           DO L=1,Nb
	      ANSWER(I,J,1,L)=0.  
	      ANSWER(I,J,2,L)=0.  
	      PP=0.  
	      QQ=0. 	
 
               DO kk=2+(L-1)*Nf,L*Nf+1
                   ANSWER(I,J,1,L) = ANSWER(I,J,1,L)
     .                        + a1(kk)*a1(kk) + b1(kk)*b1(kk)
	   	   ANSWER(I,J,2,L) = ANSWER(I,J,2,L)
     .                        + a2(kk)*a2(kk) + b2(kk)*b2(kk)
                   PP = PP + a1(kk)*a2(kk) + b1(kk)*b2(kk)
                   QQ = QQ + a2(kk)*b1(kk) - a1(kk)*b2(kk)

	       ENDDO

               !  Power spectrum for Series 1
   	      
               ANSWER(I,J,1,L) = 4.*ANSWER(I,J,1,L)/FLOAT(Nf)

               !  Power spectrum for Series 2	

               ANSWER(I,J,2,L) = 4.*ANSWER(I,J,2,L)/FLOAT(Nf)
   	      

               PP = 4.*PP/FLOAT(Nf)
   	       QQ = 4.*QQ/FLOAT(Nf)

               !  Coherence-squared in each band

	       ANSWER(I,J,7,L) = (PP*PP+QQ*QQ)
     .                     / (ANSWER(I,J,1,L)*ANSWER(I,J,2,L))

                !  Phase in each band 

	        IF(QQ.EQ.0..AND.PP.EQ.0.)THEN
	           ANSWER(I,J,8,L)=0.
	        ELSE
	           ANSWER(I,J,8,L)=ATAN2(-QQ,PP)
	        ENDIF

                ! Compute significant coherence in each band 
	        ! (currently constant for constant Nf)

                ANSWER(I,J,9,L)=Coh2sig 

                ! Compute Confidence limits for the phase
                ! theta = +- phase error (rad) at P significance 
	        !         with 2*Nf degrees of freedom
	        ! (P.285 Koopmans .The Spectral Analysis of Time Series)
                
                IF(ANSWER(I,J,7,L).EQ.0.)THEN
                  theta=3.14159              
                ELSE
                  theta = ASIN(  SQRT(  (1.-ANSWER(I,J,7,L)) / 
     .                                    ANSWER(I,J,7,L) / 
     .                                     FLOAT(2*Nf-2)   )
     .                * GetStudentT(NINT(P),Nf)   )  
                ENDIF
                ANSWER (I,J,10,L) = ANSWER(I,J,8,L)+theta ! Upper confidence level for phase
                ANSWER (I,J,11,L) = ANSWER(I,J,8,L)-theta ! Lower confidence level for phase

                ! Confidence levels on the autospectrum 

                ANSWER (I,J,3,L) = upper*ANSWER(I,J,1,L) ! Series 1: Upper confidence level	
                ANSWER (I,J,4,L) = lower*ANSWER(I,J,1,L) ! Series 1: Lower confidence level	
                ANSWER (I,J,5,L) = upper*ANSWER(I,J,2,L) ! Series 2: Upper confidence level	
                ANSWER (I,J,6,L) = lower*ANSWER(I,J,2,L) ! Series 2: Lower confidence level	

                PRINT*, 'END BAND AVERAGE'

	   ENDDO

           j1 = j1 + arg_incr(Y_AXIS,ARG1)
           j2 = j2 + arg_incr(Y_AXIS,ARG2) 
         ENDDO
         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
      ENDDO
      END


	!******************************************
	!
	!  Confidence intervals for the power spectral estimates
	!
	!******************************************
	! 
        ! Nf = number of frequencys in band	(input)
	! P  = percent level for significance (eg 90,95 0r 99 ) INPUT
	! lower = lower bound (output)
	! upper = upper bound (output)
	!
        !   IF Gxx is the 'true' autospectral value, based on the 
	!    band-averaged estimate <Gxx>, 
	!    then, for Nf number of frequencys in band,
	!    we can say with 90% certainty that
	!
        !  lower * <Gxx>  <=  Gxx  <=  upper * <Gxx> 
	!
	!  lower = df/chi squared (df,alpha/2)
	!  upper = df/chi squared (df,1-alpha/2)
	!
        ! df = degrees of freedom = 2*number of frequencys in band
	!
	! chi squared (n,alpha/2) -->  P=100*(1-alpha)
	!
	!  P  	alpha	alpha/2	 1-alpha/2
	!  90    0.1     0.05	   0.95
	!  95    0.05	 0.025	   0.975
	!  99	 0.01	 0.005	   0.995
	!
	! Bendat & Piersol, p.286
	! Bendat & Piersol, p.524 (table)

      SUBROUTINE SpectralConfidenceIntervals(Nf,P,lower,upper)
      IMPLICIT NONE
      INTEGER Nf,P,df
      REAL lower,upper
	
	!   P  	alpha	alpha/2	 1-alpha/2
	!  90    0.1     0.05	   0.95
	!  95    0.05	 0.025	   0.975
	!  99	 0.01	 0.005	   0.995
      
      REAL chi_05(15) 
      REAL chi_025(15)	  
      REAL chi_005(15)
      REAL chi_95(15)	  
      REAL chi_975(15)  
      REAL chi_995(15)  
	
      DATA chi_005 /10.6,14.86,18.55,21.96,25.19,28.3,31.3,34.27,37.16,
     .               40.,42.8,45.56,48.29,50.99,53.67/		  
      DATA chi_025 /7.38,11.14,14.45,17.53,20.48,23.34,26.12,28.85,
     .               31.53,34.17,36.78,39.36,41.92,44.46,46.98/	  
      DATA chi_05 /5.99,9.49,12.59,15.51,18.31,21.03,23.68,26.30,28.87,
     .             31.41,33.92,36.42,38.88,41.34,43.77/
      DATA chi_95 /.103,.711,1.64,2.73,3.94,5.23,6.57,7.96,8.39,10.85,
     .             12.34,13.85,15.38,16.93,18.49/
      DATA chi_975 /.0506,.484,1.24,2.18,3.25,4.4,5.63,6.91,8.23,9.59,
     .              10.98,12.4,13.84,15.31,16.79/
      DATA chi_995 /.01,.207,.676,1.34,2.16,3.07,4.07,5.14,6.26,7.43,
     .              8.64,9.89,11.16,12.46,13.79/

       df=Nf*2

	IF(df.GT.100)THEN
	  IF(P.EQ.90)THEN 
	    lower=FLOAT(df)/146.57
          upper=FLOAT(df)/95.7
        ELSEIF(P.EQ.95)THEN
		lower=FLOAT(df)/152.21
          upper=FLOAT(df)/91.58
	  ELSE  !  P.EQ.99
	  	lower=FLOAT(df)/163.65
          upper=FLOAT(df)/83.85
	  ENDIF
	ELSEIF(df.GT.50)THEN
	  IF(P.EQ.90)THEN
        	lower=FLOAT(df)/79.08
          upper=FLOAT(df)/43.19
        ELSEIF(P.EQ.95)THEN
	   	lower=FLOAT(df)/83.8
          upper=FLOAT(df)/40.48
	  ELSE  !  P.EQ.99
	   	lower=FLOAT(df)/91.95
          upper=FLOAT(df)/35.53
	  ENDIF
	ELSEIF(df.GT.30)THEN
	  IF(P.EQ.90)THEN
	   	lower=FLOAT(df)/55.76
          upper=FLOAT(df)/26.51
        ELSEIF(P.EQ.95)THEN
	   	lower=FLOAT(df)/59.34
          upper=FLOAT(df)/24.43
	  ELSE  !  P.EQ.99
	   	lower=FLOAT(df)/66.77
          upper=FLOAT(df)/20.71
	  ENDIF
	ELSE ! df <= 30
	  IF(P.EQ.90)THEN
	   	lower=FLOAT(df)/chi_05(Nf)
          upper=FLOAT(df)/chi_95(Nf)
        ELSEIF(P.EQ.95)THEN
	   	lower=FLOAT(df)/chi_025(Nf)
          upper=FLOAT(df)/chi_975(Nf)
	  ELSE  !  P.EQ.99
	   	lower=FLOAT(df)/chi_005(Nf)
          upper=FLOAT(df)/chi_995(Nf)
	  ENDIF
	ENDIF
	END

	!******************************************
	! 
	! Get the upper cutoff point of the Student T distribution 
	! (P.341 Koopmans .The Spectral Analysis of Time Series)
	!
	!******************************************
	   
	REAL FUNCTION GetStudentT(P,Nf)
        IMPLICIT NONE
	INTEGER Nf,P,df  
	REAL ST_90(15)
	REAL ST_95(15)
	REAL ST_99(15)
	REAL T
	DATA ST_90/2.92,2.132,1.943,1.86,1.812,1.782,1.761,1.746,
     .           1.734,1.725,1.717,1.711,1.706,1.701,1.697/
	DATA ST_95/4.303,2.776,2.447,2.306,2.228,2.179,2.145,2.12,
     .           2.101,2.086,2.074,2.064,2.056,2.048,2.042/
  	DATA ST_99/9.925,4.604,3.707,3.355,3.169,3.055,2.977,2.921,
     .           2.878,2.845,2.819,2.797,2.779,2.763,2.75/

	IF(P.NE.90.AND.P.NE.95.AND.P.NE.99)THEN
           P=90 
	   WRITE(6,*)'WARNING::P must be 90,95 or 99' 
           WRITE(6,*)'         in SUBROUTINE PhaseConfidenceLimits'
           WRITE(6,*)'  '
           WRITE(6,*)'P set to 90%'
	ENDIF
 
        df=Nf*2

	IF(df.GT.120)THEN
     	  IF(P.EQ.90)THEN
		T=1.645
          ELSEIF(P.EQ.95)THEN
		T=1.96
	  ELSE  !  P.EQ.99
		T=2.576
	  ENDIF
	ELSEIF(df.GT.80)THEN
	  IF(P.EQ.90)THEN
		T=1.658
          ELSEIF(P.EQ.95)THEN
		T=1.98
	  ELSE  !  P.EQ.99
		T=2.617
	  ENDIF
	ELSEIF(df.GT.50)THEN
	  IF(P.EQ.90)THEN
		T=1.671
          ELSEIF(P.EQ.95)THEN
		T=2.
	  ELSE  !  P.EQ.99
		T=2.66
	  ENDIF
	ELSEIF(df.GT.30)THEN
	  IF(P.EQ.90)THEN
		T=1.684
          ELSEIF(P.EQ.95)THEN
		T=2.021
	  ELSE  !  P.EQ.99
		T=2.704
	  ENDIF
	ELSE ! df <= 30
	  IF(P.EQ.90)THEN
	    T=ST_90(Nf)
        ELSEIF(P.EQ.95)THEN
	    T=ST_95(Nf)
	  ELSE  !  P.EQ.99
	    T=ST_99(Nf)
	  ENDIF
	ENDIF
	GetStudentT=T
	END


        !******************************************
	!
	! FFT OF A REAL PERIODIC SEQUENCE
	! 
	!******************************************
     	!
        !   N         = number of data points (input)
	!   X(N)      = real input time series	(input)
	!   a(N),b(N) = normalized Fourier coefficients (output)
	!
	!  Spectral estimates are:  S(k) = a(k) + i b(k)
	!
	!                                     i = sqrt(-1)
	!
	!
	!            S(k)=THE SUM FROM n=1,...,N OF
	!  
        !                X(n)/N*EXP(-2*PI*i*k*n/N)
	!		   so:
	!                a(k) = (1/N) * THE SUM FROM n=1,...,N OF
	!		              X(n) * cos(2*PI*k*n/N) 
	!	
	!		 b(k) = (1/N) * THE SUM FROM n=1,...,N OF  
	!                           - X(n) * sin(2*PI*k*n/N)	
	!
	! T = N dt  is the fundamental period
	! 
        ! Frequencies are: f(k) = (k-1)/T  ;  k=1,..,N
	! 
	! Frequency interval between spectral estimates: 
	!
	!            df = 1/T  = 1/(N dt)
	!
	! Nyquist frequency is at: 
	!
	!            Nyquist = 1/(2dt) = N/(2T); k = N/2 + 1
	!
	!           So:  S[1] = mean of the time series
	!                S[2] = S[N]
	!                S[3] = S[N-1]
	!                S[k] = S[N-k+1]
	!
	!           Note: S[(N/2+1] is set = 0
	!               
	SUBROUTINE FFTreal(Nin,X,a,b,WSAVE,C)
	IMPLICIT NONE
	INTEGER Nin,N,i
	REAL X(*),a(*),b(*),WSAVE(*)
	COMPLEX C(*)

        ! Check if N is even

	IF(MOD(Nin,2).EQ.0)THEN
	  N=Nin
	ELSE
	  N=Nin-1
	ENDIF
	
	! load complex array

	DO i=1,N
	  C(i)=CMPLX(X(i),0.)
	ENDDO

	! Compute the transform

	CALL CFFTF(N,C,WSAVE)

	! load and normalize the a and b arrays

	DO i=1,N
	  a(i)=REAL(C(i))/FLOAT(N)
	  b(i)=AIMAG(C(i))/FLOAT(N)
	ENDDO
	  
	! Set values at the nyquist to zero 

	a(N/2+1)=0.
	b(N/2+1)=0.

	END

C *********************************************************************
C *********************************************************************
C *********************************************************************
C *********************************************************************
C *********************************************************************


	!******************************************
	!
	!  Canned FFT Routines
	!
	!******************************************

      subroutine cfftf (n,c,wsave)

C  FORWARD COMPLEX DISCRETE FFT OF A COMPLEX PERIODIC SEQUENCE
C
C     EQUIVALENTLY, CFFTF COMPUTES THE FOURIER COEFFICIENTS 
C     OF A COMPLEX PERIODIC SEQUENCE.
C
C     THE TRANSFORM IS NOT NORMALIZED. TO OBTAIN A NORMALIZED TRANSFORM
C     THE OUTPUT MUST BE DIVIDED BY N. OTHERWISE A CALL OF CFFTF
C     FOLLOWED BY A CALL OF CFFTB WILL MULTIPLY THE SEQUENCE BY N.
C
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE CFFTF MUST BE
C     INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE).
C
C     INPUT PARAMETERS
C
C
C     N      THE LENGTH OF THE COMPLEX SEQUENCE C. THE METHOD IS
C            MORE EFFICIENT WHEN N IS THE PRODUCT OF SMALL PRIMES. N
C
C     C      A COMPLEX ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C
C     WSAVE   A REAL WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4N+1
C             IN THE PROGRAM THAT CALLS CFFTF. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY CFFTF AND CFFTB.
C
C     OUTPUT PARAMETERS
C
C     C      FOR J=1,...,N
C
C                C(J)=THE SUM FROM K=1,...,N OF
C
C                      C(K)*EXP(-I*J*K*2*PI/N)
C
C                            WHERE I=SQRT(-1)
C
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE
C             DESTROYED BETWEEN CALLS OF SUBROUTINE CFFTF OR CFFTB
C
      real       c(*)       ,wsave(*)

* INTERNAL VARIABLE DECLARATIONS

      integer n, iw1, iw2

      if (n .eq. 1) return
      iw1 = n+n+1
      iw2 = iw1+n+n
      call cfftf1 (n,c,wsave,wsave(iw1),wsave(iw2))
      end

      subroutine cfftf1 (n,c,ch,wa,ifac)

C***REFER TO CFFTF
C***ROUTINES CALLED  PASSF,PASSF5,PASSF3,PASSF2,PASSF4

      real       ch(*)      ,c(*)       ,wa(*)      ,ifac(*)

* INTERNAL VARIABLE DECLARATIONS

      INTEGER i, n, nf, na, l1, iw, k1, ip, l2, ido, idot, idl1, 
     .        ix2, ix3, ix4, nac, n2

      nf = ifac(2)
      na = 0
      l1 = 1
      iw = 1
      do 116 k1=1,nf
         ip = ifac(k1+2)
         l2 = ip*l1
         ido = n/l2
         idot = ido+ido
         idl1 = idot*l1
         if (ip .ne. 4) go to 103
         ix2 = iw+idot
         ix3 = ix2+idot
         if (na .ne. 0) go to 101
         call passf4 (idot,l1,c,ch,wa(iw),wa(ix2),wa(ix3))
         go to 102
  101    call passf4 (idot,l1,ch,c,wa(iw),wa(ix2),wa(ix3))
  102    na = 1-na
         go to 115
  103    if (ip .ne. 2) go to 106
         if (na .ne. 0) go to 104
         call passf2 (idot,l1,c,ch,wa(iw))
         go to 105
  104    call passf2 (idot,l1,ch,c,wa(iw))
  105    na = 1-na
         go to 115
  106    if (ip .ne. 3) go to 109
         ix2 = iw+idot
         if (na .ne. 0) go to 107
         call passf3 (idot,l1,c,ch,wa(iw),wa(ix2))
         go to 108
  107    call passf3 (idot,l1,ch,c,wa(iw),wa(ix2))
  108    na = 1-na
         go to 115
  109    if (ip .ne. 5) go to 112
         ix2 = iw+idot
         ix3 = ix2+idot
         ix4 = ix3+idot
         if (na .ne. 0) go to 110
         call passf5 (idot,l1,c,ch,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 111
  110    call passf5 (idot,l1,ch,c,wa(iw),wa(ix2),wa(ix3),wa(ix4))
  111    na = 1-na
         go to 115
  112    if (na .ne. 0) go to 113
         call passf (nac,idot,ip,l1,idl1,c,c,c,ch,ch,wa(iw))
         go to 114
  113    call passf (nac,idot,ip,l1,idl1,ch,ch,ch,c,c,wa(iw))
  114    if (nac .ne. 0) na = 1-na
  115    l1 = l2
         iw = iw+(ip-1)*idot
  116 continue
      if (na .eq. 0) return
      n2 = n+n
      do 117 i=1,n2
         c(i) = ch(i)
  117 continue
      end

      subroutine cffti (n,wsave)

C   INITIALIZE FOR CFFTF AND CFFTB
C     *****************************************************************
C
C     SUBROUTINE CFFTI(N,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE CFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH CFFTF AND CFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4*N+15
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH CFFTF AND CFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF CFFTF OR CFFTB
C

      real       wsave(*)

* INTERNAL VARIABLE DECLARATIONS

      integer  n, iw1, iw2

      if (n .eq. 1) return
      iw1 = n+n+1
      iw2 = iw1+n+n
      call cffti1 (n,wsave(iw1),wsave(iw2))
      end

      subroutine cffti1 (n,wa,ifac)
C***REFER TO CFFTI
C***ROUTINES CALLED  (NONE)

      real       wa(*)      ,ifac(*)    ,ntryh(4)

* INTERNAL VARIABLE DECLARATIONS

      integer n, nl, nf, j, ntry, nq, nr, i, ib, l1, k1, ip, ld, l2, 
     .        ido, idot, ipm,i1,ii
      real    argh, wldc, wlds, arg, dl1c, dl1s, dc, ds, tpi, wldch

      data ntryh(1),ntryh(2),ntryh(3),ntryh(4)/3,4,2,5/
      nl = n
      nf = 0
      j = 0
  101 j = j+1
      if (j-4) 102,102,103
  102 ntry = ntryh(j)
      go to 104
  103 ntry = ntry+2
  104 nq = nl/ntry
      nr = nl-ntry*nq
      if (nr) 101,105,101
  105 nf = nf+1
      ifac(nf+2) = ntry
      nl = nq
      if (ntry .ne. 2) go to 107
      if (nf .eq. 1) go to 107
      do 106 i=2,nf
         ib = nf-i+2
         ifac(ib+2) = ifac(ib+1)
  106 continue
      ifac(3) = 2
  107 if (nl .ne. 1) go to 104
      ifac(1) = n
      ifac(2) = nf
      tpi = 6.28318530717959
      argh = tpi/float(n)
      i = 2
      l1 = 1
      do 110 k1=1,nf
         ip = ifac(k1+2)
         ld = 0
         l2 = l1*ip
         ido = n/l2
         idot = ido+ido+2
         ipm = ip-1
         wldc = 1.
         wlds = 0.
         arg = float(l1)*argh
         dl1c = cos(arg)
         dl1s = sin(arg)
         do 109 j=1,ipm
            i1 = i
            wa(i-1) = 1.
            wa(i) = 0.
            ld = ld+l1
            wldch = wldc
            wldc = dl1c*wldc-dl1s*wlds
            wlds = dl1s*wldch+dl1c*wlds
            dc = wldc
            ds = wlds
            do 108 ii=4,idot,2
               i = i+2
               wa(i-1) = dc*wa(i-3)-ds*wa(i-2)
               wa(i) = ds*wa(i-3)+dc*wa(i-2)
  108       continue
            if (ip .le. 5) go to 109
            wa(i1-1) = wa(i-1)
            wa(i1) = wa(i)
  109    continue
         l1 = l2
  110 continue
      end


      subroutine cfftb (n,c,wsave)

C INVERSE FFT OF A COMPLEX PERIODIC SEQUENCE
C     *****************************************************************
C
C     SUBROUTINE CFFTB(N,C,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE CFFTB COMPUTES THE BACKWARD COMPLEX DISCRETE FOURIER
C     TRANSFORM (THE FOURIER SYNTHESIS). EQUIVALENTLY , CFFTB COMPUTES
C     A COMPLEX PERIODIC SEQUENCE FROM ITS FOURIER COEFFICIENTS.
C     THE TRANSFORM IS DEFINED BELOW AT OUTPUT PARAMETER C.
C
C     A CALL OF CFFTF FOLLOWED BY A CALL OF CFFTB WILL MULTIPLY THE
C     SEQUENCE BY N.
C
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE CFFTB MUST BE
C     INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE).
C
C     INPUT PARAMETERS
C
C
C     N      THE LENGTH OF THE COMPLEX SEQUENCE C. THE METHOD IS
C            MORE EFFICIENT WHEN N IS THE PRODUCT OF SMALL PRIMES.
C
C     C      A COMPLEX ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C
C     WSAVE   A REAL WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4N+1
C             IN THE PROGRAM THAT CALLS CFFTB. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY CFFTF AND CFFTB.
C
C     OUTPUT PARAMETERS
C
C     C      FOR J=1,...,N
C
C                C(J)=THE SUM FROM K=1,...,N OF
C
C                      C(K)*EXP(I*J*K*2*PI/N)
C
C                            WHERE I=SQRT(-1)
C
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE
C             DESTROYED BETWEEN CALLS OF SUBROUTINE CFFTF OR CFFTB

C***ROUTINES CALLED  CFFTB1

      real c(*), wsave(*)

* INTERNAL VARIABLE DECLARATIONS

      integer n, iw1, iw2
      if (n .eq. 1) return
      iw1 = n+n+1
      iw2 = iw1+n+n
      call cfftb1 (n,c,wsave,wsave(iw1),wsave(iw2))
      return
      end

      subroutine cfftb1 (n,c,ch,wa,ifac)

C***REFER TO CFFTB
C***ROUTINES CALLED  PASSB,PASSB5,PASSB3,PASSB2,PASSB4

      real       ch(*)      ,c(*)       ,wa(*)      ,ifac(*)

* INTERNAL VARIABLE DECLARATIONS

      integer n, nf, na, l1, iw, k1, ip, l2, ido, idot, idl1,
     .        ix2, ix3, ix4, nac, n2, i

      nf = ifac(2)
      na = 0
      l1 = 1
      iw = 1
      do 116 k1=1,nf
         ip = ifac(k1+2)
         l2 = ip*l1
         ido = n/l2
         idot = ido+ido
         idl1 = idot*l1
         if (ip .ne. 4) go to 103
         ix2 = iw+idot
         ix3 = ix2+idot
         if (na .ne. 0) go to 101
         call passb4 (idot,l1,c,ch,wa(iw),wa(ix2),wa(ix3))
         go to 102
  101    call passb4 (idot,l1,ch,c,wa(iw),wa(ix2),wa(ix3))
  102    na = 1-na
         go to 115
  103    if (ip .ne. 2) go to 106
         if (na .ne. 0) go to 104
         call passb2 (idot,l1,c,ch,wa(iw))
         go to 105
  104    call passb2 (idot,l1,ch,c,wa(iw))
  105    na = 1-na
         go to 115
  106    if (ip .ne. 3) go to 109
         ix2 = iw+idot
         if (na .ne. 0) go to 107
         call passb3 (idot,l1,c,ch,wa(iw),wa(ix2))
         go to 108
  107    call passb3 (idot,l1,ch,c,wa(iw),wa(ix2))
  108    na = 1-na
         go to 115
  109    if (ip .ne. 5) go to 112
         ix2 = iw+idot
         ix3 = ix2+idot
         ix4 = ix3+idot
         if (na .ne. 0) go to 110
         call passb5 (idot,l1,c,ch,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 111
  110    call passb5 (idot,l1,ch,c,wa(iw),wa(ix2),wa(ix3),wa(ix4))
  111    na = 1-na
         go to 115
  112    if (na .ne. 0) go to 113
         call passb (nac,idot,ip,l1,idl1,c,c,c,ch,ch,wa(iw))
         go to 114
  113    call passb (nac,idot,ip,l1,idl1,ch,ch,ch,c,c,wa(iw))
  114    if (nac .ne. 0) na = 1-na
  115    l1 = l2
         iw = iw+(ip-1)*idot
  116 continue
      if (na .eq. 0) return
      n2 = n+n
      do 117 i=1,n2
         c(i) = ch(i)
  117 continue
      end

      subroutine passb (nac,ido,ip,l1,idl1,cc,c1,c2,ch,ch2,wa)

C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)

      real       ch(ido,l1,ip)          ,cc(ido,ip,l1)          ,
     1                c1(ido,l1,ip)          ,wa(*)      ,c2(idl1,ip),
     2                ch2(idl1,ip)

* INTERNAL VARIABLE DECLARATIONS

      integer nac, ido, ip, l1, idl1, idot, nt, ipp2, ipph, idp, j, 
     .        jc, k, i, idl, inc, l, lc, ik, idlj, idij, idj
      real    war, wai

      idot = ido/2
      nt = ip*idl1
      ipp2 = ip+2
      ipph = (ip+1)/2
      idp = ip*ido
C
      if (ido .lt. l1) go to 106
      do 103 j=2,ipph
         jc = ipp2-j
         do 102 k=1,l1
            do 101 i=1,ido
               ch(i,k,j) = cc(i,j,k)+cc(i,jc,k)
               ch(i,k,jc) = cc(i,j,k)-cc(i,jc,k)
  101       continue
  102    continue
  103 continue
      do 105 k=1,l1
         do 104 i=1,ido
            ch(i,k,1) = cc(i,1,k)
  104    continue
  105 continue
      go to 112
  106 do 109 j=2,ipph
         jc = ipp2-j
         do 108 i=1,ido
            do 107 k=1,l1
               ch(i,k,j) = cc(i,j,k)+cc(i,jc,k)
               ch(i,k,jc) = cc(i,j,k)-cc(i,jc,k)
  107       continue
  108    continue
  109 continue
      do 111 i=1,ido
         do 110 k=1,l1
            ch(i,k,1) = cc(i,1,k)
  110    continue
  111 continue
  112 idl = 2-ido
      inc = 0
      do 116 l=2,ipph
         lc = ipp2-l
         idl = idl+ido
         do 113 ik=1,idl1
            c2(ik,l) = ch2(ik,1)+wa(idl-1)*ch2(ik,2)
            c2(ik,lc) = wa(idl)*ch2(ik,ip)
  113    continue
         idlj = idl
         inc = inc+ido
         do 115 j=3,ipph
            jc = ipp2-j
            idlj = idlj+inc
            if (idlj .gt. idp) idlj = idlj-idp
            war = wa(idlj-1)
            wai = wa(idlj)
            do 114 ik=1,idl1
               c2(ik,l) = c2(ik,l)+war*ch2(ik,j)
               c2(ik,lc) = c2(ik,lc)+wai*ch2(ik,jc)
  114       continue
  115    continue
  116 continue
      do 118 j=2,ipph
         do 117 ik=1,idl1
            ch2(ik,1) = ch2(ik,1)+ch2(ik,j)
  117    continue
  118 continue
      do 120 j=2,ipph
         jc = ipp2-j
         do 119 ik=2,idl1,2
            ch2(ik-1,j) = c2(ik-1,j)-c2(ik,jc)
            ch2(ik-1,jc) = c2(ik-1,j)+c2(ik,jc)
            ch2(ik,j) = c2(ik,j)+c2(ik-1,jc)
            ch2(ik,jc) = c2(ik,j)-c2(ik-1,jc)
  119    continue
  120 continue
      nac = 1
      if (ido .eq. 2) return
      nac = 0
      do 121 ik=1,idl1
         c2(ik,1) = ch2(ik,1)
  121 continue
      do 123 j=2,ip
         do 122 k=1,l1
            c1(1,k,j) = ch(1,k,j)
            c1(2,k,j) = ch(2,k,j)
  122    continue
  123 continue
      if (idot .gt. l1) go to 127
      idij = 0
      do 126 j=2,ip
         idij = idij+2
         do 125 i=4,ido,2
            idij = idij+2
            do 124 k=1,l1
               c1(i-1,k,j) = wa(idij-1)*ch(i-1,k,j)-wa(idij)*ch(i,k,j)
               c1(i,k,j) = wa(idij-1)*ch(i,k,j)+wa(idij)*ch(i-1,k,j)
  124       continue
  125    continue
  126 continue
      return
  127 idj = 2-ido
      do 130 j=2,ip
         idj = idj+ido
         do 129 k=1,l1
            idij = idj
            do 128 i=4,ido,2
               idij = idij+2
               c1(i-1,k,j) = wa(idij-1)*ch(i-1,k,j)-wa(idij)*ch(i,k,j)
               c1(i,k,j) = wa(idij-1)*ch(i,k,j)+wa(idij)*ch(i-1,k,j)
  128       continue
  129    continue
  130 continue
      end

      subroutine passb2 (ido,l1,cc,ch,wa1)

C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)

      real       cc(ido,2,l1)           ,ch(ido,l1,2)           ,
     1                wa1(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real    tr2, ti2

      if (ido .gt. 2) go to 102
      do 101 k=1,l1
         ch(1,k,1) = cc(1,1,k)+cc(1,2,k)
         ch(1,k,2) = cc(1,1,k)-cc(1,2,k)
         ch(2,k,1) = cc(2,1,k)+cc(2,2,k)
         ch(2,k,2) = cc(2,1,k)-cc(2,2,k)
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            ch(i-1,k,1) = cc(i-1,1,k)+cc(i-1,2,k)
            tr2 = cc(i-1,1,k)-cc(i-1,2,k)
            ch(i,k,1) = cc(i,1,k)+cc(i,2,k)
            ti2 = cc(i,1,k)-cc(i,2,k)
            ch(i,k,2) = wa1(i-1)*ti2+wa1(i)*tr2
            ch(i-1,k,2) = wa1(i-1)*tr2-wa1(i)*ti2
  103    continue
  104 continue
      end

      subroutine passb3 (ido,l1,cc,ch,wa1,wa2)

C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)

      real       cc(ido,3,l1)           ,ch(ido,l1,3)           ,
     1                wa1(*)     ,wa2(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real taur, taui, tr2, cr2, ti2, ci2, cr3, ci3, dr2, dr3, di2, di3

      data taur,taui /-.5,.866025403784439/
      if (ido .ne. 2) go to 102
      do 101 k=1,l1
         tr2 = cc(1,2,k)+cc(1,3,k)
         cr2 = cc(1,1,k)+taur*tr2
         ch(1,k,1) = cc(1,1,k)+tr2
         ti2 = cc(2,2,k)+cc(2,3,k)
         ci2 = cc(2,1,k)+taur*ti2
         ch(2,k,1) = cc(2,1,k)+ti2
         cr3 = taui*(cc(1,2,k)-cc(1,3,k))
         ci3 = taui*(cc(2,2,k)-cc(2,3,k))
         ch(1,k,2) = cr2-ci3
         ch(1,k,3) = cr2+ci3
         ch(2,k,2) = ci2+cr3
         ch(2,k,3) = ci2-cr3
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            tr2 = cc(i-1,2,k)+cc(i-1,3,k)
            cr2 = cc(i-1,1,k)+taur*tr2
            ch(i-1,k,1) = cc(i-1,1,k)+tr2
            ti2 = cc(i,2,k)+cc(i,3,k)
            ci2 = cc(i,1,k)+taur*ti2
            ch(i,k,1) = cc(i,1,k)+ti2
            cr3 = taui*(cc(i-1,2,k)-cc(i-1,3,k))
            ci3 = taui*(cc(i,2,k)-cc(i,3,k))
            dr2 = cr2-ci3
            dr3 = cr2+ci3
            di2 = ci2+cr3
            di3 = ci2-cr3
            ch(i,k,2) = wa1(i-1)*di2+wa1(i)*dr2
            ch(i-1,k,2) = wa1(i-1)*dr2-wa1(i)*di2
            ch(i,k,3) = wa2(i-1)*di3+wa2(i)*dr3
            ch(i-1,k,3) = wa2(i-1)*dr3-wa2(i)*di3
  103    continue
  104 continue
      end

      subroutine passb4 (ido,l1,cc,ch,wa1,wa2,wa3)

C***REFER TO CFFTB
C***ROUTINES CALLED (NONE)

      real       cc(ido,4,l1)           ,ch(ido,l1,4)           ,
     1                wa1(*)     ,wa2(*)     ,wa3(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real    ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4, 
     .        cr2, cr3, cr4, ci2, ci3, ci4

      if (ido .ne. 2) go to 102
      do 101 k=1,l1
         ti1 = cc(2,1,k)-cc(2,3,k)
         ti2 = cc(2,1,k)+cc(2,3,k)
         tr4 = cc(2,4,k)-cc(2,2,k)
         ti3 = cc(2,2,k)+cc(2,4,k)
         tr1 = cc(1,1,k)-cc(1,3,k)
         tr2 = cc(1,1,k)+cc(1,3,k)
         ti4 = cc(1,2,k)-cc(1,4,k)
         tr3 = cc(1,2,k)+cc(1,4,k)
         ch(1,k,1) = tr2+tr3
         ch(1,k,3) = tr2-tr3
         ch(2,k,1) = ti2+ti3
         ch(2,k,3) = ti2-ti3
         ch(1,k,2) = tr1+tr4
         ch(1,k,4) = tr1-tr4
         ch(2,k,2) = ti1+ti4
         ch(2,k,4) = ti1-ti4
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            ti1 = cc(i,1,k)-cc(i,3,k)
            ti2 = cc(i,1,k)+cc(i,3,k)
            ti3 = cc(i,2,k)+cc(i,4,k)
            tr4 = cc(i,4,k)-cc(i,2,k)
            tr1 = cc(i-1,1,k)-cc(i-1,3,k)
            tr2 = cc(i-1,1,k)+cc(i-1,3,k)
            ti4 = cc(i-1,2,k)-cc(i-1,4,k)
            tr3 = cc(i-1,2,k)+cc(i-1,4,k)
            ch(i-1,k,1) = tr2+tr3
            cr3 = tr2-tr3
            ch(i,k,1) = ti2+ti3
            ci3 = ti2-ti3
            cr2 = tr1+tr4
            cr4 = tr1-tr4
            ci2 = ti1+ti4
            ci4 = ti1-ti4
            ch(i-1,k,2) = wa1(i-1)*cr2-wa1(i)*ci2
            ch(i,k,2) = wa1(i-1)*ci2+wa1(i)*cr2
            ch(i-1,k,3) = wa2(i-1)*cr3-wa2(i)*ci3
            ch(i,k,3) = wa2(i-1)*ci3+wa2(i)*cr3
            ch(i-1,k,4) = wa3(i-1)*cr4-wa3(i)*ci4
            ch(i,k,4) = wa3(i-1)*ci4+wa3(i)*cr4
  103    continue
  104 continue
      end

      subroutine passb5 (ido,l1,cc,ch,wa1,wa2,wa3,wa4)

C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)

      real       cc(ido,5,l1)           ,ch(ido,l1,5)           ,
     1                wa1(*)     ,wa2(*)     ,wa3(*)     ,wa4(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real tr2, tr4, tr5, tr11, tr12, ti2, ti3, ti4, ti5, ti11, ti12, 
     .     cr2, cr3, cr4, cr5, ci2, ci3, ci4, ci5, dr4, dr5, 
     .     di2, di3, di4, di5, tr3, dr3, dr2

      data tr11,ti11,tr12,ti12 /.309016994374947,.951056516295154,
     1-.809016994374947,.587785252292473/
      if (ido .ne. 2) go to 102
      do 101 k=1,l1
         ti5 = cc(2,2,k)-cc(2,5,k)
         ti2 = cc(2,2,k)+cc(2,5,k)
         ti4 = cc(2,3,k)-cc(2,4,k)
         ti3 = cc(2,3,k)+cc(2,4,k)
         tr5 = cc(1,2,k)-cc(1,5,k)
         tr2 = cc(1,2,k)+cc(1,5,k)
         tr4 = cc(1,3,k)-cc(1,4,k)
         tr3 = cc(1,3,k)+cc(1,4,k)
         ch(1,k,1) = cc(1,1,k)+tr2+tr3
         ch(2,k,1) = cc(2,1,k)+ti2+ti3
         cr2 = cc(1,1,k)+tr11*tr2+tr12*tr3
         ci2 = cc(2,1,k)+tr11*ti2+tr12*ti3
         cr3 = cc(1,1,k)+tr12*tr2+tr11*tr3
         ci3 = cc(2,1,k)+tr12*ti2+tr11*ti3
         cr5 = ti11*tr5+ti12*tr4
         ci5 = ti11*ti5+ti12*ti4
         cr4 = ti12*tr5-ti11*tr4
         ci4 = ti12*ti5-ti11*ti4
         ch(1,k,2) = cr2-ci5
         ch(1,k,5) = cr2+ci5
         ch(2,k,2) = ci2+cr5
         ch(2,k,3) = ci3+cr4
         ch(1,k,3) = cr3-ci4
         ch(1,k,4) = cr3+ci4
         ch(2,k,4) = ci3-cr4
         ch(2,k,5) = ci2-cr5
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            ti5 = cc(i,2,k)-cc(i,5,k)
            ti2 = cc(i,2,k)+cc(i,5,k)
            ti4 = cc(i,3,k)-cc(i,4,k)
            ti3 = cc(i,3,k)+cc(i,4,k)
            tr5 = cc(i-1,2,k)-cc(i-1,5,k)
            tr2 = cc(i-1,2,k)+cc(i-1,5,k)
            tr4 = cc(i-1,3,k)-cc(i-1,4,k)
            tr3 = cc(i-1,3,k)+cc(i-1,4,k)
            ch(i-1,k,1) = cc(i-1,1,k)+tr2+tr3
            ch(i,k,1) = cc(i,1,k)+ti2+ti3
            cr2 = cc(i-1,1,k)+tr11*tr2+tr12*tr3
            ci2 = cc(i,1,k)+tr11*ti2+tr12*ti3
            cr3 = cc(i-1,1,k)+tr12*tr2+tr11*tr3
            ci3 = cc(i,1,k)+tr12*ti2+tr11*ti3
            cr5 = ti11*tr5+ti12*tr4
            ci5 = ti11*ti5+ti12*ti4
            cr4 = ti12*tr5-ti11*tr4
            ci4 = ti12*ti5-ti11*ti4
            dr3 = cr3-ci4
            dr4 = cr3+ci4
            di3 = ci3+cr4
            di4 = ci3-cr4
            dr5 = cr2+ci5
            dr2 = cr2-ci5
            di5 = ci2-cr5
            di2 = ci2+cr5
            ch(i-1,k,2) = wa1(i-1)*dr2-wa1(i)*di2
            ch(i,k,2) = wa1(i-1)*di2+wa1(i)*dr2
            ch(i-1,k,3) = wa2(i-1)*dr3-wa2(i)*di3
            ch(i,k,3) = wa2(i-1)*di3+wa2(i)*dr3
            ch(i-1,k,4) = wa3(i-1)*dr4-wa3(i)*di4
            ch(i,k,4) = wa3(i-1)*di4+wa3(i)*dr4
            ch(i-1,k,5) = wa4(i-1)*dr5-wa4(i)*di5
            ch(i,k,5) = wa4(i-1)*di5+wa4(i)*dr5
  103    continue
  104 continue
      end

      subroutine passf (nac,ido,ip,l1,idl1,cc,c1,c2,ch,ch2,wa)

C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)

      real       ch(ido,l1,ip)          ,cc(ido,ip,l1)          ,
     1                c1(ido,l1,ip)          ,wa(*)      ,c2(idl1,ip),
     2                ch2(idl1,ip)

* INTERNAL VARIABLE DECLARATIONS

      integer nac, ido, ip, l1, idl1, idot, nt, ipp2, ipph, idp, j, jc,
     .        k, i, idl, inc, l, lc, ik, idlj, idij, idj
      real    war, wai

      idot = ido/2
      nt = ip*idl1
      ipp2 = ip+2
      ipph = (ip+1)/2
      idp = ip*ido
C
      if (ido .lt. l1) go to 106
      do 103 j=2,ipph
         jc = ipp2-j
         do 102 k=1,l1
            do 101 i=1,ido
               ch(i,k,j) = cc(i,j,k)+cc(i,jc,k)
               ch(i,k,jc) = cc(i,j,k)-cc(i,jc,k)
  101       continue
  102    continue
  103 continue
      do 105 k=1,l1
         do 104 i=1,ido
            ch(i,k,1) = cc(i,1,k)
  104    continue
  105 continue
      go to 112
  106 do 109 j=2,ipph
         jc = ipp2-j
         do 108 i=1,ido
            do 107 k=1,l1
               ch(i,k,j) = cc(i,j,k)+cc(i,jc,k)
               ch(i,k,jc) = cc(i,j,k)-cc(i,jc,k)
  107       continue
  108    continue
  109 continue
      do 111 i=1,ido
         do 110 k=1,l1
            ch(i,k,1) = cc(i,1,k)
  110    continue
  111 continue
  112 idl = 2-ido
      inc = 0
      do 116 l=2,ipph
         lc = ipp2-l
         idl = idl+ido
         do 113 ik=1,idl1
            c2(ik,l) = ch2(ik,1)+wa(idl-1)*ch2(ik,2)
            c2(ik,lc) = -wa(idl)*ch2(ik,ip)
  113    continue
         idlj = idl
         inc = inc+ido
         do 115 j=3,ipph
            jc = ipp2-j
            idlj = idlj+inc
            if (idlj .gt. idp) idlj = idlj-idp
            war = wa(idlj-1)
            wai = wa(idlj)
            do 114 ik=1,idl1
               c2(ik,l) = c2(ik,l)+war*ch2(ik,j)
               c2(ik,lc) = c2(ik,lc)-wai*ch2(ik,jc)
  114       continue
  115    continue
  116 continue
      do 118 j=2,ipph
         do 117 ik=1,idl1
            ch2(ik,1) = ch2(ik,1)+ch2(ik,j)
  117    continue
  118 continue
      do 120 j=2,ipph
         jc = ipp2-j
         do 119 ik=2,idl1,2
            ch2(ik-1,j) = c2(ik-1,j)-c2(ik,jc)
            ch2(ik-1,jc) = c2(ik-1,j)+c2(ik,jc)
            ch2(ik,j) = c2(ik,j)+c2(ik-1,jc)
            ch2(ik,jc) = c2(ik,j)-c2(ik-1,jc)
  119    continue
  120 continue
      nac = 1
      if (ido .eq. 2) return
      nac = 0
      do 121 ik=1,idl1
         c2(ik,1) = ch2(ik,1)
  121 continue
      do 123 j=2,ip
         do 122 k=1,l1
            c1(1,k,j) = ch(1,k,j)
            c1(2,k,j) = ch(2,k,j)
  122    continue
  123 continue
      if (idot .gt. l1) go to 127
      idij = 0
      do 126 j=2,ip
         idij = idij+2
         do 125 i=4,ido,2
            idij = idij+2
            do 124 k=1,l1
               c1(i-1,k,j) = wa(idij-1)*ch(i-1,k,j)+wa(idij)*ch(i,k,j)
               c1(i,k,j) = wa(idij-1)*ch(i,k,j)-wa(idij)*ch(i-1,k,j)
  124       continue
  125    continue
  126 continue
      return
  127 idj = 2-ido
      do 130 j=2,ip
         idj = idj+ido
         do 129 k=1,l1
            idij = idj
            do 128 i=4,ido,2
               idij = idij+2
               c1(i-1,k,j) = wa(idij-1)*ch(i-1,k,j)+wa(idij)*ch(i,k,j)
               c1(i,k,j) = wa(idij-1)*ch(i,k,j)-wa(idij)*ch(i-1,k,j)
  128       continue
  129    continue
  130 continue
      end

      subroutine passf2 (ido,l1,cc,ch,wa1)

C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)

      real       cc(ido,2,l1)           ,ch(ido,l1,2)           ,
     1                wa1(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real    tr2, ti2

      if (ido .gt. 2) go to 102
      do 101 k=1,l1
         ch(1,k,1) = cc(1,1,k)+cc(1,2,k)
         ch(1,k,2) = cc(1,1,k)-cc(1,2,k)
         ch(2,k,1) = cc(2,1,k)+cc(2,2,k)
         ch(2,k,2) = cc(2,1,k)-cc(2,2,k)
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            ch(i-1,k,1) = cc(i-1,1,k)+cc(i-1,2,k)
            tr2 = cc(i-1,1,k)-cc(i-1,2,k)
            ch(i,k,1) = cc(i,1,k)+cc(i,2,k)
            ti2 = cc(i,1,k)-cc(i,2,k)
            ch(i,k,2) = wa1(i-1)*ti2-wa1(i)*tr2
            ch(i-1,k,2) = wa1(i-1)*tr2+wa1(i)*ti2
  103    continue
  104 continue
      end

      subroutine passf3 (ido,l1,cc,ch,wa1,wa2)

C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)

      real       cc(ido,3,l1)           ,ch(ido,l1,3)           ,
     1                wa1(*)     ,wa2(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real taur, taui, tr2, cr2, ti2, ci2, cr3, ci3, dr2, dr3, di2, di3

      data taur,taui /-.5,-.866025403784439/
      if (ido .ne. 2) go to 102
      do 101 k=1,l1
         tr2 = cc(1,2,k)+cc(1,3,k)
         cr2 = cc(1,1,k)+taur*tr2
         ch(1,k,1) = cc(1,1,k)+tr2
         ti2 = cc(2,2,k)+cc(2,3,k)
         ci2 = cc(2,1,k)+taur*ti2
         ch(2,k,1) = cc(2,1,k)+ti2
         cr3 = taui*(cc(1,2,k)-cc(1,3,k))
         ci3 = taui*(cc(2,2,k)-cc(2,3,k))
         ch(1,k,2) = cr2-ci3
         ch(1,k,3) = cr2+ci3
         ch(2,k,2) = ci2+cr3
         ch(2,k,3) = ci2-cr3
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            tr2 = cc(i-1,2,k)+cc(i-1,3,k)
            cr2 = cc(i-1,1,k)+taur*tr2
            ch(i-1,k,1) = cc(i-1,1,k)+tr2
            ti2 = cc(i,2,k)+cc(i,3,k)
            ci2 = cc(i,1,k)+taur*ti2
            ch(i,k,1) = cc(i,1,k)+ti2
            cr3 = taui*(cc(i-1,2,k)-cc(i-1,3,k))
            ci3 = taui*(cc(i,2,k)-cc(i,3,k))
            dr2 = cr2-ci3
            dr3 = cr2+ci3
            di2 = ci2+cr3
            di3 = ci2-cr3
            ch(i,k,2) = wa1(i-1)*di2-wa1(i)*dr2
            ch(i-1,k,2) = wa1(i-1)*dr2+wa1(i)*di2
            ch(i,k,3) = wa2(i-1)*di3-wa2(i)*dr3
            ch(i-1,k,3) = wa2(i-1)*dr3+wa2(i)*di3
  103    continue
  104 continue
      end

      subroutine passf4 (ido,l1,cc,ch,wa1,wa2,wa3)

C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)

      real       cc(ido,4,l1)           ,ch(ido,l1,4)           ,
     1                wa1(*)     ,wa2(*)     ,wa3(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real    ti1, ti2, tr4, ti3, tr1, tr2, ti4, tr3, cr3, ci3, 
     .        cr2, cr4, ci2, ci4

      if (ido .ne. 2) go to 102
      do 101 k=1,l1
         ti1 = cc(2,1,k)-cc(2,3,k)
         ti2 = cc(2,1,k)+cc(2,3,k)
         tr4 = cc(2,2,k)-cc(2,4,k)
         ti3 = cc(2,2,k)+cc(2,4,k)
         tr1 = cc(1,1,k)-cc(1,3,k)
         tr2 = cc(1,1,k)+cc(1,3,k)
         ti4 = cc(1,4,k)-cc(1,2,k)
         tr3 = cc(1,2,k)+cc(1,4,k)
         ch(1,k,1) = tr2+tr3
         ch(1,k,3) = tr2-tr3
         ch(2,k,1) = ti2+ti3
         ch(2,k,3) = ti2-ti3
         ch(1,k,2) = tr1+tr4
         ch(1,k,4) = tr1-tr4
         ch(2,k,2) = ti1+ti4
         ch(2,k,4) = ti1-ti4
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            ti1 = cc(i,1,k)-cc(i,3,k)
            ti2 = cc(i,1,k)+cc(i,3,k)
            ti3 = cc(i,2,k)+cc(i,4,k)
            tr4 = cc(i,2,k)-cc(i,4,k)
            tr1 = cc(i-1,1,k)-cc(i-1,3,k)
            tr2 = cc(i-1,1,k)+cc(i-1,3,k)
            ti4 = cc(i-1,4,k)-cc(i-1,2,k)
            tr3 = cc(i-1,2,k)+cc(i-1,4,k)
            ch(i-1,k,1) = tr2+tr3
            cr3 = tr2-tr3
            ch(i,k,1) = ti2+ti3
            ci3 = ti2-ti3
            cr2 = tr1+tr4
            cr4 = tr1-tr4
            ci2 = ti1+ti4
            ci4 = ti1-ti4
            ch(i-1,k,2) = wa1(i-1)*cr2+wa1(i)*ci2
            ch(i,k,2) = wa1(i-1)*ci2-wa1(i)*cr2
            ch(i-1,k,3) = wa2(i-1)*cr3+wa2(i)*ci3
            ch(i,k,3) = wa2(i-1)*ci3-wa2(i)*cr3
            ch(i-1,k,4) = wa3(i-1)*cr4+wa3(i)*ci4
            ch(i,k,4) = wa3(i-1)*ci4-wa3(i)*cr4
  103    continue
  104 continue
      end

      subroutine passf5 (ido,l1,cc,ch,wa1,wa2,wa3,wa4)

C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)

      real       cc(ido,5,l1)           ,ch(ido,l1,5)           ,
     1                wa1(*)     ,wa2(*)     ,wa3(*)     ,wa4(*)

* INTERNAL VARIABLE DECLARATIONS

      integer ido, l1, k, i
      real    tr11, ti11, tr12, ti12, ti5, ti2, ti4, ti3, tr5, tr2, 
     .        tr4, tr3, cr2, ci2, cr3, ci3, cr5, ci5, cr4, ci4, dr3, 
     .        dr4, di3, di4, dr5, dr2, di5, di2

      data tr11,ti11,tr12,ti12 /.309016994374947,-.951056516295154,
     1-.809016994374947,-.587785252292473/
      if (ido .ne. 2) go to 102
      do 101 k=1,l1
         ti5 = cc(2,2,k)-cc(2,5,k)
         ti2 = cc(2,2,k)+cc(2,5,k)
         ti4 = cc(2,3,k)-cc(2,4,k)
         ti3 = cc(2,3,k)+cc(2,4,k)
         tr5 = cc(1,2,k)-cc(1,5,k)
         tr2 = cc(1,2,k)+cc(1,5,k)
         tr4 = cc(1,3,k)-cc(1,4,k)
         tr3 = cc(1,3,k)+cc(1,4,k)
         ch(1,k,1) = cc(1,1,k)+tr2+tr3
         ch(2,k,1) = cc(2,1,k)+ti2+ti3
         cr2 = cc(1,1,k)+tr11*tr2+tr12*tr3
         ci2 = cc(2,1,k)+tr11*ti2+tr12*ti3
         cr3 = cc(1,1,k)+tr12*tr2+tr11*tr3
         ci3 = cc(2,1,k)+tr12*ti2+tr11*ti3
         cr5 = ti11*tr5+ti12*tr4
         ci5 = ti11*ti5+ti12*ti4
         cr4 = ti12*tr5-ti11*tr4
         ci4 = ti12*ti5-ti11*ti4
         ch(1,k,2) = cr2-ci5
         ch(1,k,5) = cr2+ci5
         ch(2,k,2) = ci2+cr5
         ch(2,k,3) = ci3+cr4
         ch(1,k,3) = cr3-ci4
         ch(1,k,4) = cr3+ci4
         ch(2,k,4) = ci3-cr4
         ch(2,k,5) = ci2-cr5
  101 continue
      return
  102 do 104 k=1,l1
         do 103 i=2,ido,2
            ti5 = cc(i,2,k)-cc(i,5,k)
            ti2 = cc(i,2,k)+cc(i,5,k)
            ti4 = cc(i,3,k)-cc(i,4,k)
            ti3 = cc(i,3,k)+cc(i,4,k)
            tr5 = cc(i-1,2,k)-cc(i-1,5,k)
            tr2 = cc(i-1,2,k)+cc(i-1,5,k)
            tr4 = cc(i-1,3,k)-cc(i-1,4,k)
            tr3 = cc(i-1,3,k)+cc(i-1,4,k)
            ch(i-1,k,1) = cc(i-1,1,k)+tr2+tr3
            ch(i,k,1) = cc(i,1,k)+ti2+ti3
            cr2 = cc(i-1,1,k)+tr11*tr2+tr12*tr3
            ci2 = cc(i,1,k)+tr11*ti2+tr12*ti3
            cr3 = cc(i-1,1,k)+tr12*tr2+tr11*tr3
            ci3 = cc(i,1,k)+tr12*ti2+tr11*ti3
            cr5 = ti11*tr5+ti12*tr4
            ci5 = ti11*ti5+ti12*ti4
            cr4 = ti12*tr5-ti11*tr4
            ci4 = ti12*ti5-ti11*ti4
            dr3 = cr3-ci4
            dr4 = cr3+ci4
            di3 = ci3+cr4
            di4 = ci3-cr4
            dr5 = cr2+ci5
            dr2 = cr2-ci5
            di5 = ci2-cr5
            di2 = ci2+cr5
            ch(i-1,k,2) = wa1(i-1)*dr2+wa1(i)*di2
            ch(i,k,2) = wa1(i-1)*di2-wa1(i)*dr2
            ch(i-1,k,3) = wa2(i-1)*dr3+wa2(i)*di3
            ch(i,k,3) = wa2(i-1)*di3-wa2(i)*dr3
            ch(i-1,k,4) = wa3(i-1)*dr4+wa3(i)*di4
            ch(i,k,4) = wa3(i-1)*di4-wa3(i)*dr4
            ch(i-1,k,5) = wa4(i-1)*dr5+wa4(i)*di5
            ch(i,k,5) = wa4(i-1)*di5-wa4(i)*dr5
  103    continue
  104 continue
      end 
