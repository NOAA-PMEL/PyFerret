	SUBROUTINE IS_READ ( memory, *, *, status )

* read a grid of data from a system file
* this routine merely looks at the class of data set to decide on the
* appropriate underlying routine to actually get the data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/19/87
* revision 0.1 - 9/11/87 - changed check of ds_format to ds_type 
*			   and 'GT' --> '  GT' for new TMAP library
* revision 0.2 - 11/2/88 - IS_READ_GT --> IS_READ_TM (ss-symmetrical)
* V200: 10/25/89 - EZ data type added
* 	 2/27/90 - unspecified limits are set with FLESH_OUT_AXIS
* V230:  3/2/92 - added netCDF support
* V312: 5/94 - array "memory" as a calling argument
* V320: 2/95 - allow for category cat_file_rev (a file variable with
*		reversed axes)
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA statment rather than a
*	        PARAMETER statement because variable was modified in code
* 	      - Added a preprocessor include for tmap_dset.parm
*	        because it needed a preprocessor.
* calling argument declarations:
	INTEGER	status
	REAL	memory(*)
* normal exit is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

* internal variable declarations:
	LOGICAL		reprimand
	DATA       	reprimand/.TRUE./
	INTEGER 	CGRID_AXIS, cx, dset_num, idim, cat
	CHARACTER	VAR_CODE*8

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
#	include	"tmap_format/tmap_dset.parm"
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* --- end of introductory code ---

* initialize
	cx = is_cx( isp )
	dset_num = cx_data_set( cx )

* for the duration of this routine replace cat_revfile_var with cat_file_var
* (since most routines cannot cope with cat_revfile_var)
	cat = cx_category(cx)
	cx_category(cx) = cat_file_var

* ensure that all needed limits are known
	DO 100 idim = 1, 4
	   IF ( cx_lo_ss( cx, idim ) .NE. unspecified_int4 ) GOTO 100
	   IF ( CGRID_AXIS( idim, cx ) .EQ. mnormal ) GOTO 100
	   IF ( reprimand ) THEN
	      CALL WARN
     .		('FERRET is more efficient if regions are fully specified' )
	      CALL WARN( ww_dim_name(idim)//
     .		' axis limits are unspecified - possibly others, too' )
	      reprimand = .FALSE.		! only remind them once
	   ENDIF
	   CALL FLESH_OUT_AXIS( idim, cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000		! should never happen
!	   CALL ERRMSG( ferr_limits, status,
!     .			ww_dim_name(idim)//' axis limits are unknown for '//
!     .			VAR_CODE( cat_file_var, cx_variable(cx) ), *5000 )
 100	CONTINUE

* choose the appropriate routine to read the data
	IF ( ds_type( dset_num ) .EQ. pds_type_ez ) THEN
	   CALL READ_EZ( memory, cx, is_mr(isp), status )
	   IF ( status .EQ. ferr_ok ) GOTO 2
	ELSE
	   CALL READ_TM( memory, cx, is_mr(isp), status )
	   IF ( status .EQ. ferr_ok ) GOTO 2
	ENDIF

* error exit
 5000	cx_category(cx) = cat
 	RETURN

* never need components since TM IO library handles 4D accesses as of 11/88
! 1	cx_category(cx) = cat
!	RETURN 1

* done - we got what was asked for
 2	cx_category(cx) = cat
	RETURN 2
	END	
