      SUBROUTINE GET_VIEW_SIZE( vxsize, vysize )

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
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xplot_state.cmn'
	include	'tmap_pplv11inc/plt_inc.decl'	! with sizing data,etc.
	include	'pplv11inc/PLT.INC'	! with sizing data, etc.
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

