	SUBROUTINE GCF_GET_AXIS_SRCS(ifcn, iarg, axis_source)

* return instructions to Ferret on how this particular argument will
* get merged into the joint context of a grid-changing function

* V450: 1/97
*	7/97 - slight reorg for external functions
*       8/97 - type mismatch on axis_implied_from caught by 
*              linux/aix compilers *kob*
* calling argument declarations
	INTEGER	ifcn, iarg, axis_source(4)

* internal variable declarations
	INTEGER	idim, iptr1,
     .		axis_will_be(4)
	LOGICAL axis_implied_from(4)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
	include 'ferret_cmn/xfr_grid.cmn'	! w/ mgrid_abstract
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'	! W/ MGRID_ABSTRACT
#endif

* initialize
* is this a valid argument number?
	IF (iarg .LT. 1
     . .OR. iarg .GT. gfcn_num_reqd_args(ifcn) )
     .					STOP 'gcf_cx_merge_arg_1'

* store info in local arrays to blur distinction between internal and external
	IF ( ifcn .LE. gfcn_num_internal ) THEN
* ... determine pointer to argument information
	   iptr1 = gfcn_arg_ptr(ifcn)
	   DO 10 idim = 1, 4
	      axis_will_be(idim) = gfcn_axis_will_be(idim,ifcn)
	      axis_implied_from(idim) =
     .			gfcn_axis_implied_from(idim,iptr1+iarg-1)
 10	   CONTINUE
	ELSE
	   STOP 'gcf_get_axis_src: call for external function'
	ENDIF

* return information about argument iarg
*    axis_source may be returned as
*	pgc_impose_axis
*	pgc_merge_axis
*    or	pgc_ignore_axis

	DO 100 idim = 1, 4

	   IF (axis_will_be(idim) .EQ. pgc_axis_is_abstract)	THEN
	      axis_source(idim) = pgc_impose_axis

	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_is_normal) THEN
	      axis_source(idim) = pgc_impose_axis

	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein) THEN
	      axis_source(idim) = pgc_impose_axis

	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_implied_by_args) THEN

	      IF (axis_implied_from(idim)) THEN
	         axis_source(idim) = pgc_merge_axis
	      ELSE
	         axis_source(idim) = pgc_ignore_axis
	      ENDIF

	   ELSE
	      STOP 'Invalid axis merging information in GC func'

	   ENDIF

 100	CONTINUE

	RETURN
	END
