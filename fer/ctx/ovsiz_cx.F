	SUBROUTINE OVSIZ_CX( cx )

* expand the context to grab more data than currently indicated

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/26/89

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	LOGICAL	ovsiz, user_var
	INTEGER	grid, idim, ndim, pdim, dim(4), ss, ii, status, var
	REAL*8	TM_WORLD

* initialize
	user_var = cx_category(cx) .EQ. cat_user_var
	var	 = cx_variable(cx)
	grid	 = cx_grid(cx)
	CALL GET_CX_DIMS( cx, ndim, dim )

* can only perform over-sizing on a plane
* cannot rely on ndim. e.g.: CONTOUR/k=lo:hi/i=lo:hi/j=lo:hi NEW
*			where DEFINE VARIABLE NEW = U[K=@AVE]
	pdim = 0
	DO 100 ii = 1, ndim
	   idim = dim(ii)
	   IF ( user_var ) THEN
	      IF ( uvar_given(idim,var) .EQ. uvlim_needed ) pdim = pdim + 1
	   ELSE
	      pdim = pdim + 1
	   ENDIF
 100	CONTINUE
	IF ( pdim .NE. 2 ) GOTO 1000

	DO 200 ii = 1, ndim
	   idim = dim(ii)

* cannot resize safely if limits are given WITHIN user-var def'n
	   IF ( user_var ) THEN
	      IF ( uvar_given(idim,var) .NE. uvlim_needed ) GOTO 200
	   ENDIF

* assume resizing isn't needed until proven otherwise
	   ovsiz = .FALSE.	
* check lower limit
	   ss = cx_lo_ss( cx, idim )
	   IF ( cx_lo_ww(idim,cx) .LT. TM_WORLD(ss,grid,idim,box_middle) )
     .									  THEN
	        cx_lo_ss( cx, idim ) = ss - 1
	        ovsiz = .TRUE.
	   ENDIF
* check upper limit
	   ss = cx_hi_ss( cx, idim )
	   IF ( cx_hi_ww(idim,cx) .GT. TM_WORLD(ss,grid,idim,box_middle) )
     .									  THEN
	        cx_hi_ss( cx, idim ) = ss + 1
	        ovsiz = .TRUE.
	   ENDIF
* re-flesh axis if necessary
	   IF ( ovsiz ) THEN
	      cx_by_ss(idim,cx) = .TRUE.
	      CALL CONFINE_AXIS  ( idim, cx, status )	! ignore status
	      CALL FLESH_OUT_AXIS( idim, cx, status )	! ignore status
	   ENDIF
 200	CONTINUE

* flag that no further oversizing is needed
 1000	cx_ovsiz(cx) = ovsiz_off

	RETURN
	END
