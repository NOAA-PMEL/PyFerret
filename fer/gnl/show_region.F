	SUBROUTINE SHOW_REGION( cx )

* execute SHOW REGION for a single context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/23/87
* revision 0.1 - 3/23/88 - enlarged cx_DIM_STR to accomodate date/time strings
* V200:  5/19/89 - reordered lo/hi_ss args
* Unix/RISC port - 2/27/91 - mystery error with version of FORTRAN compiler
*		          eliminated using temporary buffer for cx_dim_str
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	INTEGER	cx

* local parameter declarations:
	LOGICAL	    full_precision
	PARAMETER ( full_precision = .TRUE. )

* local variable declarations:
	INTEGER  slen, idim
	CHARACTER cx_DIM_STR*48

* write region name
	IF ( cx .EQ. cx_last ) THEN
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				'default region:',15)
	ELSE
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				'region '//cx_name(cx)//':',12)
	ENDIF

* describe each axis - produce lines as complex as:"X=lo:hi@xxx:nn (transform)"
	DO 100 idim = 1, 4
* axis unspecified ?
	   IF (  cx_by_ss(idim,cx)
     .	        .AND. cx_lo_ss (cx,idim) .EQ. unspecified_int4
     .	   .OR..NOT.cx_by_ss(idim,cx)
     .		.AND.  cx_lo_ww (idim,cx) .EQ. unspecified_val8) THEN
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				'        '//ww_dim_name(idim)//
     .				'/'//ss_dim_name(idim)//' is unspecified',
     .				26 )
	   ELSE

* write it out
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun,'        '
     .		//CX_DIM_STR(idim,cx,':',full_precision,slen),0)

	   ENDIF

 100	CONTINUE	      

	RETURN
	END
