	SUBROUTINE T_DERIV( com, mcom, res, mres, dt, reg )

* compute centered derivative of component along axis t_dim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/1/89 - based on DO_DERIV_CNTR
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xunits.cmn_text'
	external xunits_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:
	LOGICAL reg
	INTEGER	mcom, mres
	REAL	   dt,
     .		com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	INTEGER i, j, k, l, grid, lo_l, hi_l, lm1, lp1
	REAL	bad_com, bad_res, lo_com, hi_com, box, box1
	REAL*8	TM_WORLD

* initialize
	grid    = mr_grid( mres )
	bad_com = mr_bad_data( mres )
	bad_res = mr_bad_data( mcom )

* fill in edges if not computable
	lo_l = mr_lo_s4(mres)
	IF ( mr_lo_s4(mcom) .EQ. mr_lo_s4(mres) ) THEN
	   DO 100 l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO 100 k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO 100 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 100 i = mr_lo_s1(mres), mr_hi_s1(mres)
 100	   res(i,j,k,lo_l) = bad_res
	   lo_l = lo_l + 1
	ENDIF
	hi_l = mr_hi_s4(mres)
	IF ( mr_hi_s4(mcom) .EQ. mr_hi_s4(mres) ) THEN
	   DO 200 l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO 200 k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO 200 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 200 i = mr_lo_s1(mres), mr_hi_s1(mres)
 200	   res(i,j,k,hi_l) = bad_res
	   hi_l = hi_l - 1
	ENDIF
	   
* loop through all computable points
	DO 1000 l = lo_l, hi_l

	lm1 = l-1
	lp1 = l+1

* time box size
	   box = TM_WORLD( lp1, grid, t_dim, box_middle )
     .	       - TM_WORLD( lm1, grid, t_dim, box_middle )

* all T spacings the same ?
	   IF ( l .EQ. lo_l ) THEN
	      box1 = box
	      reg  = .TRUE.
	   ELSE
	      IF ( box .NE. box1 ) reg = .FALSE.
	   ENDIF

	   DO 500 k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO 500 j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO 500 i = mr_lo_s1(mres), mr_hi_s1(mres)

	      lo_com = com(i,j,k,lm1)
	      hi_com = com(i,j,k,lp1)
	      IF ( lo_com.EQ.bad_res .OR. hi_com.EQ.bad_res ) THEN
	         res(i,j,k,l) = bad_res
	      ELSE
	         res(i,j,k,l) = (hi_com-lo_com) / box
	      ENDIF

 500	   CONTINUE
 1000	CONTINUE

* compute delta t
	IF ( reg ) THEN
	   dt = box
	ELSE
	   box = TM_WORLD( hi_l+1, grid, t_dim, box_middle )
     .	       - TM_WORLD( lo_l-1, grid, t_dim, box_middle )
	   dt = box / (hi_l-lo_l+1)
	ENDIF

	RETURN
	END	
