      REAL FUNCTION THETA_FOFF (S, TO, PO, PR)

**************************

C TO COMPUTE LOCAL POTENTIAL TEMPERATURE AT PF 
C USING BRYDEN 1973 POLYNOMIAL FOR ADIABATIC LAPSE RATE
C AND RUNGE-KUTTA 4-TH ORDER INTEGRATION ALGORITHM.
C
C REFERENCES:	BRYDEN,H.,1973,DEEP-SEA RES.,20,401-408
C		FOFONOFF,N,M,1977,DEEP-SEA RES.,24,489-491
C
C UNITS:
C	PRESSURE	PO	DECIBARS
C	TEMPERATURE	TO	DEG CELSIUS (IPTS-68)
C	SALINITY	S	(PSS-78)
C	REFERENCE PRS	PR	DECIBARS
C	POTENTIAL TEMP	THETA	DEG CELSIUS
C
C CHECKVALUE: THETA=36.89073 DEG C:  S=40(PSS-78), TO=40 DEG C,
C		PO=10000 DECIBARS, PR=0 DECIBARS
C
C--FROM FOFONOFF,N.,AND R.C.MILLARD,1983,DRAFT. LJM

*************************

* calling argument declarations:
      REAL S, TO, PO, PR

* internal variable declarations:
      REAL ATG, T, p, h, xk, q

      P = PO
      T = TO
      H = PR - P
      XK = H*ATG(S,T,P) 
      T = T + 0.5*XK
      Q = XK
      P = P + 0.5*H 
      XK = H*ATG(S,T,P) 
      T = T + 0.29289322*(XK - Q) 
      Q = 0.58578644*XK + 0.121320344*Q 
      XK = H*ATG(S,T,P) 
      T = T + 1.707106781*(XK - Q)
      Q = 3.414213562*XK - 4.121320344*Q
      P = P + 0.5*H 
      XK = H*ATG(S,T,P) 
      THETA_FOFF = T + (XK - 2.0*Q)/6.0
      RETURN
      END 

* **************************************************
        REAL FUNCTION ATG (salinity, temperature, pressure) 
    
*    Reference:  Bryden, H., 1973, Deep Sea Res., 20, 401-408.
*    
*    Units:
*    
*    	Pressure:		P			decibars
*    	Temperature:		T			degrees Celcius
*    	Salinity:		S			psu (IPSS-78)
*    	Adiabatic:		ATG			deg. C/decibar
*    	
*    	
*    Check value:
*    		ATG = 3.255796E-4 C/dbar for S = 40 (psu), T = 40 deg. C, 
*    		P = 10000 decibars.

* calling argument declarations:
        REAL salinity, temperature, pressure

* internal variable declarations:
	REAL S, T, p
	
* initialize
	S = salinity - 35.
	T = temperature
	p = pressure
	
	ATG = (((-2.1687e-16 * T + 1.8676e-14) * T - 4.6206e-13) * p
     .	      + ((2.7759e-12 * T - 1.1351e-10) * S + ((-5.4481e-14 * T
     .	      + 8.733e-12) * T - 6.7795e-10) * T + 1.8741e-08)) * p
     .	      + (-4.2393e-08 * T + 1.8932e-06) * S
     .	      + ((6.6228e-10 * T - 6.836e-08) * T + 8.5258e-06) * T
     .	      + 3.5803e-05
	
        RETURN
        END
