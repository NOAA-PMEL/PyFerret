	SUBROUTINE TEK4014_LINE_BUNDLE (zwsid)

*	Routine to set up the line bundles for tek4014's
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks

	integer		zwsid,i,j
#ifdef atc_gks
        integer         ltype(6)
	data		ltype /1,2,3,4,-1001,-1004/
#else
	integer 	ltype(5)
	data		ltype /1,2,3,4,-3/
#endif
	real		thickness

*******************************************************************************

#ifdef atc_gks
*	INCREASE THICKNESS AND REPEAT LINETYPE EACH SIX INDICES
100     do 20 j = 1,3
	   thickness = real (j)*2.0 - 1.0

           do 10 i = 1,6
              call gsplr (zwsid,i+6*(j-1),ltype(i),thickness,1)
10        continue
20     continue
#else
*	INCREASE THICKNESS AND REPEAT LINETYPE EACH FIVE INDICES
	do 20 j = 1,3
	   thickness = real (j)*2.0 - 1.0

	   do 10 i = 1,5
	      call gsplr (zwsid,i+5*(j-1),ltype(i),thickness,1)
10	   continue
20	continue
#endif
	return

	end
