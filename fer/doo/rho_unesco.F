        REAL FUNCTION RHO_UNESCO(salinity, temperature, pressure)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* *sh* 8/5/92 - adapted from C version supplied by John Osborne, PMEL

*       This calculates rho (density kg/m^3) at salinity, 
*	temperature, and pressure.  This will be potential density if 
*	temperature is potential temperature.  The routine uses the high 
*	pressure equation of state from Millero et al. 1980 and the one-
*	atmosphere equation of state from Millero and Poisson 1981 as
*	reported in Gill 1982.  The notation follows Millero et al. 1980
*	and Millero and Poisson 1981.
*	
*	Note: the routine takes p in decibars and converts to bars for
*	the calculations.
*	
* References:	Millero, F.J., et al., 1980, Deep-Sea Res., 27A, 255-264.
*  Millero, F.J. and Alain Poisson, 1981, Deep-Sea Res., 28A, 625-629.
*  Gill, A.E., 1982, Atmosphere-Ocean Dynamics, Academic Press, Inc., 662 pp. 
*	
*	Input Units:
*			S: psu	T: deg. C	p: decibars.
*			
*	Output Units:
*			rho: kg/m^3
*			
*	Check values:
*			rho = 999.96675 	for S = 0,  T = 5,  p = 0
*			rho = 1027.675465 	for S = 35, T = 5,  p = 0
*			rho = 1062.538172 	for S = 35, T = 25, p = 10000.
			
* calling argument declaraions:
        REAL salinity, temperature, pressure

* internal variable declarations:
	REAL S, T, p, rootS
	REAL a, b, c, d, e
	REAL Aw, Bw, Kw
	REAL A2, B2, C2, Kzero, K
	REAL rhoW, rhoZero
	
* initialize
	S = salinity
	T = temperature
	p = pressure / 10.
	rootS = SQRT (S)
		
* rhoW is the density of pure water at temperature T.
	rhoW = ((((6.536332e-09 * T - 1.120083e-06) * T + 1.001685e-04)
     .		* T - 9.095290e-03) * T + 6.793952e-02) * T + 999.842594
	
	A2 = (((5.3875e-09 * T - 8.2467e-07) * T + 7.6438e-05)
     .		* T - 4.0899e-03) * T + 8.24493e-01
				
	B2 = (-1.6546e-06 * T + 1.0227e-04) * T - 5.72466e-03
				
	C2 = 4.8314e-04

* rhoZero is the one-atmosphere density of seawater
	rhoZero	= (C2 * S + B2 * rootS + A2) * S + rhoW
	
	IF ( pressure .EQ. 0.0 ) THEN
	   RHO_UNESCO = rhoZero
	   RETURN
        ENDIF

	a = ((-6.1670e-05 * T + 1.09987e-02) * T - 6.03459e-01) * T
     .		+ 54.6746
			
	b = (-5.3009e-04 * T + 1.6483e-02) * T + 7.944e-02
	
	c = (-1.6078e-06 * T - 1.0981e-05) * T + 2.2838e-03
	
	d = 1.91075e-04
	
	e = (9.1697e-10 * T + 2.0816e-08) * T - 9.9348e-07
	
	Aw = ((-5.77905e-07 * T + 1.16092e-04) * T + 1.43713e-03) * T
     .	       + 3.239908
			
	Bw = (5.2787e-08 * T - 6.12293e-06) * T + 8.50935e-05

* Kw is the secant bulk modulus of pure water at temperature T.	
	Kw = (((-5.155288e-05 * T + 1.360477e-02) * T - 2.327105)
     .		* T + 148.4206) * T + 19652.21

* Kzero is the secant bulk modulus of seawater at one atmosphere.
	Kzero = (b * rootS + a) * S + Kw

* K is the secant bulk modulus of seawater at (S, T, p).		
	K = ((e * p + d * rootS + c) * S + Bw * p + Aw) * p + Kzero
	
	RHO_UNESCO = rhoZero / (1.0 - (p / K))
	
	RETURN
	END
