	CHARACTER*(*) FUNCTION GCF_NAME( ifcn )

* return the name of a grid changing function from its ID number

* V450: 1/97
*       97.07.16 *jc* adding code to allow call to efcn_get_name()

* calling argument declarations
	INTEGER ifcn

* internal variable declarations
	INTEGER status

* internal parameter declarations:
        INTEGER      slen
        PARAMETER  ( slen = 40 )
 
#ifdef sun
        BYTE      fhol(slen)    ! c-type Hollerith string buffer
#else
        INTEGER*1 fhol(slen)    ! c-type Hollerith string buffer
#endif
 
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* check the internal grid changing functions first
	IF (ifcn .LT. 0 ) THEN

	   CALL ERRMSG( ferr_internal, status, 'gcf_name', *5000)

	ELSEIF (ifcn .LE. gfcn_num_internal) THEN

	   GCF_NAME = gfcn_name(ifcn)

	ELSE

	   call efcn_get_name(ifcn, fhol)
	   call TM_CTOF_STRNG( fhol, gcf_name, slen)

	ENDIF	   
	
* success
	RETURN

* error exit
 5000	GCF_NAME = 'ILLEGAL NAME'
	RETURN

	END
