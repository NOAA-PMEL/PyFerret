	SUBROUTINE SHOW_1_FUNCTION( lun, its_gc, brief, ifcn )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* output a text description about the specified function

* there are 3 classes of functions:
*  classic internal Ferret functions (non-grid-changing) e.g. SIN(X)
*  internal grid-changing functions
*  external grid changing functions

* programmer - steve hankin

* V450: 7/97
* 97.07.30 *jc* added code to enable external functions
* V500: 1/99 *sh* added support for string arguments to functions

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/grid_chg_fcns.parm'
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
	INTEGER TM_LENSTR, TM_LENSTR1, GCF_ARG_TYPE,
     .		ll, ll2, llsave, itype, narg, i, iarg, iarg2
	CHARACTER units*40, gcf_name*40, ef_arg_name*40, ef_descr*80, 
	1    ef_units*40
	integer efcn_scan, efcn_get_num_reqd_args, efcn_gather_info

* local parameter definitions
	CHARACTER*4 indent
	PARAMETER ( indent = '    ' )
        INTEGER      slen
        PARAMETER  ( slen = 80 )

#ifdef sun
        BYTE      fhol(slen)    ! c-type Hollerith string buffer
#else
        INTEGER*1 fhol(slen)    ! c-type Hollerith string buffer
#endif
 
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
	   IF ( efcn_scan(gfcn_num_internal) .EQ. 0 ) THEN
	      ! JC_TODO: what if no external functions are found?
	      STOP 'no external functions found'
	   ELSE
	      IF ( efcn_gather_info(ifcn) .ne. 0 ) THEN
		 RETURN
	      ENDIF
	      risc_buff = gcf_name(ifcn)
	      narg = efcn_get_num_reqd_args(ifcn)
	      iarg = 1
	   ENDIF
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
           call efcn_get_arg_name(ifcn, iarg, fhol)
           call TM_CTOF_STRNG( fhol, ef_arg_name, slen)
	   risc_buff = risc_buff(:ll) // ef_arg_name
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
	      call efcn_get_arg_name(ifcn, iarg2, fhol)
	      call TM_CTOF_STRNG( fhol, ef_arg_name, slen)
	      risc_buff = risc_buff(:ll) // ef_arg_name
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
	   call efcn_get_descr(ifcn, fhol)
	   call TM_CTOF_STRNG( fhol, ef_descr, slen)
	   risc_buff = ef_descr
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
	      call efcn_get_arg_name(ifcn, iarg, fhol)
	      call TM_CTOF_STRNG( fhol, ef_arg_name, slen)
	      risc_buff = ef_arg_name
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
	      call efcn_get_arg_desc(ifcn, iarg, fhol)
	      call TM_CTOF_STRNG( fhol, ef_descr, slen)
	      risc_buff = risc_buff(:ll+2) // ef_descr
	   ENDIF
* ... arg units
	   IF ( .NOT.its_gc ) THEN
	      units = alg_arg_units(iarg)
	   ELSEIF ( ifcn .LE. gfcn_num_internal ) THEN
	      units = gfcn_arg_units(iarg)
	   ELSE
	      call efcn_get_arg_unit(ifcn, iarg, fhol)
	      call TM_CTOF_STRNG( fhol, ef_units, slen)
	      units = ef_units
	   ENDIF
	   ll2 = TM_LENSTR(units)
	   IF ( ll2 .GT. 0 ) THEN
	      ll = TM_LENSTR1(risc_buff)
	      IF ( ll .GE. len_rbuff ) GOTO 400   ! truncation error
	      risc_buff = risc_buff(:ll) // ' (' //units(:ll2) //')'
	   ENDIF
	   ll = TM_LENSTR1(risc_buff)
* ... data type (if STRING)
	   IF ( .NOT.its_gc ) THEN
	      itype = parg_type_float
	   ELSE
	      itype = GCF_ARG_TYPE(ifcn, i)
	   ENDIF
	   IF (itype .EQ. parg_type_string) THEN
	       risc_buff = risc_buff(:ll) // ' (STRING)'
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


