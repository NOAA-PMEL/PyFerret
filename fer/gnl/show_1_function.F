	SUBROUTINE SHOW_1_FUNCTION( lun, its_gc, brief, ifcn )

* output a text description about the specified function

* there are 3 classes of functions:
*  classic internal Ferret functions (non-grid-changing) e.g. SIN(X)
*  internal grid-changing functions
*  external grid changing functions

* programmer - steve hankin

* V450: 7/97

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xalgebra.cmn'
	include	'ferret_cmn/xgrid_chg_fcns.cmn'
	include	'ferret_cmn/xrisc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE	'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations:
	LOGICAL	its_gc, brief
	INTEGER	lun, ifcn
	
* local variable declarations:
	INTEGER TM_LENSTR, TM_LENSTR1, ll, ll2, llsave,
     .					narg, i, iarg, iarg2
	CHARACTER units*40

* local parameter definitions
	CHARACTER*4 indent
	PARAMETER ( indent = '    ' )

* initialize
	len_rbuff = LEN(risc_buff)	! maximum length

* 1st line -- function name with args in parens
* ... name
	IF ( .NOT.its_gc ) THEN
	   risc_buff = alg_fcn(ifcn)
	   narg = alg_num_fcn_args(ifcn)
	   iarg = alg_fcn_arg_ptr(ifcn)
	ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	   risc_buff = gfcn_name(ifcn)
	   narg = gfcn_num_reqd_args(ifcn)
	   iarg = gfcn_arg_ptr(ifcn)
	ELSE
	   STOP 'no external functions yet'
	ENDIF

* ... open paren
	ll = TM_LENSTR1(risc_buff)
	risc_buff = risc_buff(:ll)//'('
	ll = ll + 1
	IF ( ll .GE. len_rbuff ) GOTO 200   ! undetected (unlikely) error

* ... first argument
	IF ( .NOT.its_gc ) THEN
	   risc_buff = risc_buff(:ll) // alg_arg_name(iarg)
	ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	   risc_buff = risc_buff(:ll) // gfcn_arg_name(iarg)
	ELSE
	   STOP 'no external functions yet'
	ENDIF
	ll = TM_LENSTR1(risc_buff)
	IF ( ll .GE. len_rbuff ) GOTO 200   ! undetected (unlikely) error

* ... additional arguments, if any
        iarg2 = iarg
	DO 100 i = 2, narg
	   iarg2 = iarg2 + 1      ! point to next arg
	   risc_buff = risc_buff(:ll) // ','
	   ll = ll + 1
	   IF ( .NOT.its_gc ) THEN
	      risc_buff = risc_buff(:ll) // alg_arg_name(iarg2)
	   ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	      risc_buff = risc_buff(:ll) // gfcn_arg_name(iarg2)
	   ELSE
	      STOP 'no external functions yet'
	   ENDIF
	   ll = TM_LENSTR1(risc_buff)
	   IF ( ll .GE. len_rbuff ) GOTO 200   ! truncation error
 100	CONTINUE

* ... closing parenthesis
	risc_buff = risc_buff(:ll)//')'
	ll = ll + 1

* ... write out the function name with its args
 200	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, ll)
	IF ( brief ) RETURN

* write 2nd line (if available) -- description of function
	IF ( .NOT.its_gc ) THEN
	   risc_buff = alg_fcn_descr(ifcn)
	ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	   risc_buff = gfcn_descr(ifcn)
	ELSE
	   STOP 'no external functions yet'
	ENDIF
	IF ( risc_buff .NE. ' ') CALL SPLIT_LIST
     .			(pttmode_explct, lun, indent//risc_buff, 0)

* write list of arguments and their definitions
* ... "argname: description (units)"
	DO 500 i = 1, narg
* ... arg name
	   IF ( .NOT.its_gc ) THEN
	      risc_buff = alg_arg_name(iarg)
	   ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	      risc_buff = gfcn_arg_name(iarg)
	   ELSE
	      STOP 'no external functions yet'
	   ENDIF
	   ll = TM_LENSTR1(risc_buff)
	   IF ( ll .GE. len_rbuff ) GOTO 400   ! truncation error
* ... arg description
	   risc_buff = risc_buff(:ll) //': '
	   llsave = ll + 1		! save length of "arg:"
	   IF ( .NOT.its_gc ) THEN
	      risc_buff = risc_buff(:ll+2) // alg_arg_descr(iarg)
	   ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	      risc_buff = risc_buff(:ll+2) // gfcn_arg_descr(iarg)
	   ELSE
	      STOP 'no external functions yet'
	   ENDIF
* ... arg units
	   IF ( .NOT.its_gc ) THEN
	      units = alg_arg_units(iarg)
	   ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	      units = gfcn_arg_units(iarg)
	   ELSE
	      STOP 'no external functions yet'
	   ENDIF
	   ll2 = TM_LENSTR(units)
	   IF ( ll2 .GT. 0 ) THEN
	      ll = TM_LENSTR1(risc_buff)
	      IF ( ll .GE. len_rbuff ) GOTO 400   ! truncation error
	      risc_buff = risc_buff(:ll) // ' (' //units(:ll2) //')'
	   ENDIF
	   ll = TM_LENSTR1(risc_buff)

* ... write the line
 400	   IF (ll .GT. llsave) CALL SPLIT_LIST
     .			(pttmode_explct, lun, indent//risc_buff, 4+ll)

* ... go back for the next argument
	   iarg = iarg + 1      ! point to next arg
 500	CONTINUE

	RETURN
	END


