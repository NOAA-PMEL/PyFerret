      SUBROUTINE GET_VIEW_SIZE( vxsize, vysize )

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
* determine the X and Y size of the current viewport in Denbo inches
* This routine is to be called directly from PLOT+.
* It requires an External declaration in the main program to ensure
* that it is loaded at link time.

* Further, this routine supports a kludge:
* PLOT+ "should" really be aware only of the current viewport.
* The ASIZE,BSIZE known by PLOT+ should be the vaules that
* this routine is computing.  Currently, PLOT+ creates the full window and
* FERRET creates and modifies windows by communicating with PLOT+.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* Unix/RISC port - 10/91
* V300:  4/19/93 - old style vp's: use ABS() for negative vp_size

#ifdef unix
	include 'ferret.parm'
	include 'xplot_state.cmn'
	include	'plt_inc.decl'	! with sizing data,etc.
	include	'PLT.INC'	! with sizing data, etc.
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE	'TMAP_PPLV11INC:PLT_INC.DECL/LIST'	! with sizing data,etc.
	INCLUDE	'PPLV11INC:PLT.INC/LIST'	! with sizing data, etc.
#endif
	
* calling argument declarations:
	REAL vxsize, vysize

* internal variable declarations:
	REAL	vxmax, vymax, scale

* initialize to full plot size
        vxsize = width
        vysize = height

* If viewports are active compute fraction of plot size in viewport
	IF ( vp_num .NE. mvp_dflt ) THEN
           IF ( vp_xclip(vp_num) .EQ. unspecified_val4 ) THEN
              vxmax = 1.0
              vymax = 1.0
           ELSE
              vxmax = vp_xclip(vp_num)
              vymax = vp_yclip(vp_num)
           ENDIF
           scale = 1.0 / SQRT( ABS(vp_size(vp_num)) )
           vxsize = scale * vxsize * (vxmax - vp_xorg(vp_num))
           vysize = scale * vysize * (vymax - vp_yorg(vp_num))
	ENDIF

	RETURN
	END

