	SUBROUTINE PS_LINE_BUNDLE (zwsid)

*	Routine to set up the line bundles for monochrome PS ws's
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks

	integer		zwsid,i,j
#ifdef atc_gks
	integer 	ltype(6)
	data   	        ltype/1,2,3,4,-2,-3/
#else
        integer         ltype(6)
        data            ltype/1,-5,3,4,-3,-1/
#endif
	real		thickness

C	For linux *jd* 12.96 put initialization in data statements
	
*******************************************************************************

	do 20 j = 1,3
	   thickness = real (j)*2.0 - 1.0

	   do 10 i = 1,6
	      call gsplr (zwsid,i+6*(j-1),ltype(i),thickness,1)
10	   continue
20	continue

	return

	end
