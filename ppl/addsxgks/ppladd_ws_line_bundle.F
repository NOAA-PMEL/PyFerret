	SUBROUTINE WS_LINE_BUNDLE (zwsid,zwstype)

*	Routine to set up the line bundles for generic WS's.
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks
*	*jd* 8.10.93 mod to support XGKS line colors -- force setting

	integer		zwsid,zwstype,i,j

#ifdef atc_gks
	integer 	ltype(6)
	data	 	ltype/1,2,3,4,-1001,-1004/
#else
	integer 	ltype(6)
	data		ltype/1,2,3,4,-3,-4/
#endif

	integer		max_pline,max_pmark,max_text,max_fill_area
	integer		max_pattern,max_color,error
	integer		fktr
                                       
	real		thickness

*******************************************************************************

*	WHAT IS AVAILABLE ON THIS WORKSTATION?
	call gqlwk (zwstype,error,max_pline,max_pmark,
     .	            max_text,max_fill_area,max_pattern,max_color)
	if (error .ne. 0) goto 1000

*	COLOR OR MONOCHROME?
	if (max_color .le. 2) goto 100

*	COLOR WORKSTATIONS:
#ifdef xgks
        call gscr (zwsid,2,1.0,0.0,0.0)
        call gscr (zwsid,3,0.0,1.0,0.0)
        call gscr (zwsid,4,0.0,0.0,1.0)
        call gscr (zwsid,5,0.0,1.0,1.0)
        call gscr (zwsid,6,1.0,0.0,1.0)
#endif

	do 20 j = 1,6 ! colors
	   do 10 fktr = 1,3
	      call gsplr (zwsid,(fktr-1)*6+j,1,real(fktr),j)
10	   continue
20	continue

	return

*	MONOCHROME WORKSTATIONS:
100	do 120 j = 1,3
	   thickness = real (j)

	   do 110 i = 1,6
	      call gsplr (zwsid,i+6*(j-1),ltype(i),thickness,1)
110	   continue
120	continue                   

	return

*	FATAL ERR
1000	return

	end
