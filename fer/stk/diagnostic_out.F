	SUBROUTINE DIAGNOSTIC_OUT( action, isub, pointer_type )

* output a line describing the activity in progress

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/5/87
* V200:  5/19/89 - re-ordered lo/hi_ss args
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420: 12/20/95 - do not print isub - changes too much between revisions
* Linux Port 12/96 *kob* - missing comma in 3000 format statement 
*                          discovered by f90 compiler
* 			 - Added a preprocessor include for tmap_dset.parm
*			   because it needed a preprocessor.
* V450: 7/7/97 - make a readable short name from VAR_CODE

* calling argument declarations:
	INTEGER		isub, pointer_type
	CHARACTER*(*)	action

* internal argument declarations:
	INTEGER	    TM_LENSTR1, lenbuff, i, dset
	CHARACTER   VAR_CODE*24, buff24*24, buff8*8

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'tmap_format/tmap_dims.parm'
#	include	"tmap_format/tmap_dset.parm"
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xrisc.cmn'		! 12/94 SPLIT_LIST buffer
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'		! 12/94 SPLIT_LIST BUFFER
#endif

	risc_buff = ' '

* readable variable name code
	IF ( pointer_type .EQ. point_to_cx ) THEN
 	   buff24 = VAR_CODE( cx_category(isub), cx_variable(isub) )
	ELSE
	   buff24 = VAR_CODE( mr_category(isub), mr_variable(isub) )
	ENDIF
	IF ( buff24(:2) .EQ. '(C' ) THEN
! ... use low order 2 digits from 3-digit numbers (shortening to 2-digits)
	   buff8 = '(C'//buff24(4:5)//',V'//buff24(9:10)
	ELSE
	   lenbuff = TM_LENSTR1(buff24)
! ... assemble short name from beginning and end of full name 
	   IF (lenbuff .GT. 8 ) THEN
	      buff8 = buff24(1:3)//'-'//buff24(lenbuff-3:lenbuff)
	   ELSE
	      buff8 = buff24
	   ENDIF
	ENDIF

	IF ( pointer_type .EQ. point_to_cx ) THEN
	   IF ( cx_data_set(isub) .EQ. unspecified_int4 ) THEN
	      dset = -9	! full -9999 won't fit in output field
	   ELSE
	      dset = cx_data_set(isub)
	   ENDIF
	   WRITE ( risc_buff,3000 )
     .			action,
     .		 	buff8,
     .			dset,
     .			(cx_lo_ss(isub,i),cx_hi_ss(isub,i),i=1,4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )
	ELSE
	   IF ( mr_category(isub) .EQ. cat_constant
     .	   .OR. mr_category(isub) .EQ. cat_pseudo_var ) THEN
	      dset = -9
	   ELSE
	      dset = mr_data_set( isub )
	   ENDIF
	   WRITE ( risc_buff,3000 )
     .			action,
     .			buff8,
     .			dset,
     .			(mr_lo_ss(isub,i),mr_hi_ss(isub,i),i=1,4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )
	ENDIF
 3000	FORMAT (1X,A,1X,A8,' D:',I2,
     .				'  I:',2I5,'  J:',2I5,'  K:',2I5,'  L:',2I5)
	RETURN
	END	
