        SUBROUTINE SET_ONE_COLOR
        
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
*       Sets the RGB values for a single color index
*       J Davison JISAO/PMEL/TMAP 11.15.91
*       *jd* 3.10.92 Brought over to Sun
*	*jd* 1.14.94 Mod for background setting in xgks
*       *js* 7.31.97 mod to change call to gqlwk to query_xgks_wkstn
* V610  *acm* 3/08 Fixes for metafile batch mode
*       *kms* 10/11 white is gks color 0 and black is gks color 1
*             (opposite of XWindows).  Since gks colors are used
*             in the graphics delegate, switch these values.
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* PyFr *kms* 5/13 Add alpha channel (opaque), allow background color (0) to be set

        IMPLICIT NONE
        include		'gkspar.inc'
	include		'cmrd_inc.decl'
        include         'CMRD.INC'
	include		'gkscm1_inc.decl'
        include         'GKSCM1.INC'
	include		'gkscm2.cmn'
	include         'parampl5_dat.decl'
	include         'PARAMPL5.DAT'
	include         'save_colors.cmn'
        INCLUDE         'pen_inc.decl'  ! with num_line_colors
        INCLUDE         'PEN.INC'

	integer		ndx,i,err
        integer         max_pline,max_pmark,max_text,max_fill_area
        integer         max_pattern,max_color
	real		val(4)

	integer		type, error
	REAL		rd, gr, bl


******************************************************************************
*	IF NO ARGS, RESET COLORS
	if (m .eq. 0) then
           call FGD_GQLWK (wstype,err,max_pline,max_pmark,
     .              max_text,max_fill_area,max_pattern,max_color)
	   if (err .ne. 0) goto 1000

	   do 5 i = 0,num_line_colors
	      call FGD_GQPCR(wstype,i,err,val(1),val(2),val(3),val(4))
	      if (err .ne. 0) goto 5
	      call FGD_GSCR(wsid,i,val(1),val(2),val(3),val(4))
5	   continue
	   goto 1000
	end if



*	ELSE SET COLOR P(1) to RGBA: P(2), P(3), P(4), P(5)
*       where p(5) is optional (default 1.0)
	if ( (m .lt. 4) .or. (m .gt. 5) ) goto 1000
	
	if (p(1) .ge. 0.0 .and. p(1) .le. 255) then
	   ndx = ifix (p(1))
	else
	   goto 1000
	end if

        val(4) = 1.0
	do 10 i = 1,m-1
	   if (p(i+1) .ge. 0.0 .and. p(i+1) .le. 100.0) then
	      val(i) = p(i+1)/100.0
	   else
	      goto 1000
	   end if
10	continue

	CALL FGD_GSCR(wsid,ndx,val(1),val(2),val(3),val(4))
*       color zero is always the color of the background for this window
	if (ndx .eq. 0)
     .     CALL FGD_REDRAW_BACKGROUND(wsid)

 1000   continue

        call FGD_GUWK(wsid,gpostp)

        return
	end


