	SUBROUTINE WS_LINE_BUNDLE (zwsid,zwstype)

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
*	Routine to set up the line bundles for generic WS's.
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks
*	*jd* 8.10.93 mod to support XGKS line colors -- force setting
*       *js* 7.31.97 mod to change call to gqlwk to query_xgks_wkstn

*       *acm* 8/2001  add polymarker call for PLOT/SYMBOL=DOT  
* gsplr: wkid, line index, line type, linewidth scale factor, color index
* gspmr: wkid, marker index, marker type(dot=1), 
*        size scale factor (const for dot), color index

*        NOTE: as of 10/3/01  this is not working consistently. With 
*        marktype=1, the documentation says the symbol will be a "point"
*        and that the mark size is ignored.  In fact the mark size is not 
*        ignored, and if I use the argument as in the /THICKNESS qualifier 
*        for lines, we get three different sizes for the marks. The marks 
*        are not round dots however.  Also, when saving the plot to a metafile, 
*        the postscript image has the dots looking much smaller than they 
*        do on the Ferret window or in a gif image.  They are so small they 
*        are invisible in ghostview except for the largest size. They do print 
*        as tiny dots.

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
	integer		fktr, marktype
                                       
	real		thickness

*******************************************************************************

*	WHAT IS AVAILABLE ON THIS WORKSTATION?
	call query_xgks_wkstn (zwstype,error,max_pline,max_pmark,
     .	            max_text,max_fill_area,max_pattern,max_color)
	if (error .ne. 0) goto 1000

        marktype = 1    ! polymarker type 1=dot, (4=circle)

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
	      call gspmr (zwsid,(fktr-1)*6+j,marktype,real(fktr),j)
10	   continue
20	continue

	return

*	MONOCHROME WORKSTATIONS:
100	do 120 j = 1,3
	   thickness = real (j)
	   do 110 i = 1,6
	      call gsplr (zwsid,i+6*(j-1),ltype(i),thickness,1)
	      call gspmr (zwsid,i+6*(j-1),marktype,real(fktr),1)
110	   continue
120	continue                   

	return

*	FATAL ERR
1000	return

	end
