	SUBROUTINE PLOTIFY_GRP_CX( cx_list, ncx )

* modify the joint context and document flags for a plot so that information
* which is implicitly documented via the plotting axes is not duplicated in
* the plot labels

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  1/5/90
* V230:   6/9/92 - added WIRE command
*        7/22/92 - turn off in-plane key info e.g. xy plane: VECT u,v[g=temp]
*  1/28/93 - for SHADE, CONTOUR and VECTOR return a 2D region if possible

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/command.parm'
	include	'ferret_cmn/slash.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xplot_setup.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:COMMAND.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations
        INTEGER ncx, cx_list(ncx)

* internal variable declarations
	INTEGER	idim, cx, icx, ndim, ifix, pdim

* for PLOT/VS and WIRE the along-axis limits do need to be displayed
        IF ( cmnd_num .EQ. cmnd_wire
     . .OR. (cmnd_num .EQ. cmnd_plot
     . .AND. qual_given(slash_plot_vs).GT.0) ) RETURN

* 1/93
* for SHADE, CONTOUR, and VECTOR promote 0 and 1D regions to 2D if possible
        IF ( cmnd_num .EQ. cmnd_contour
     . .OR.  cmnd_num .EQ. cmnd_shade
     . .OR.  cmnd_num .EQ. cmnd_vector ) THEN
           ndim = nplot_axis + 1
           DO 200 ifix = ndim, 2
              DO 150 idim = 1, 4
* axis already included in nplot_axis ?
                 DO 100 pdim = 1, nplot_axis
                    IF ( idim .EQ. plot_axis(pdim) ) GOTO 150
 100             CONTINUE
                 DO 110 icx = 1, ncx
                    cx = cx_list(icx)
* suitable dimensions must have valid ww coords and indices
                    IF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4
     .              .OR. cx_lo_ww(idim,cx) .EQ. unspecified_val8
     .              .OR. cx_lo_ww(idim,cx) .GE. cx_hi_ww(idim,cx) )
     .                                                    GOTO 150
 110             CONTINUE
* found a suitable dimension to promote the plot context with
                 plot_axis(ifix) = idim
                 nplot_axis = nplot_axis + 1
                 GOTO 200
 150          CONTINUE
 200       CONTINUE
        ENDIF

* limits along the plot axes are implicitly documented
	DO 300 pdim = 1, nplot_axis
	   idim = plot_axis(pdim)
	   cx_lo_ww(idim,cx_plot) = unspecified_val8
	   cx_hi_ww(idim,cx_plot) = unspecified_val8
	   key_doc(idim) = .FALSE.      ! 7/92
* ... but transformations along the plot axes are not
	   key_doc(idim+pdoc_offset_xform) = .TRUE.
 300	CONTINUE

	RETURN
	END
