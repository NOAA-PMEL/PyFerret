	SUBROUTINE BREAK_VARNAME( text, dset_in, varname, attname, 
     .                            varid, status )
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
* Break up a name of the form varname.attname
* Check whether the variable exists within the dataset
* stored in linked-list structure of nc datasets.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V600 6/05 *acm* Attribute control.
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* V604 6/07 *acm* Fix bug 1518 If we didnt find the varid in the dataset,
*                 it may  be a user var(as in break_varattname, where the bug
*                 occurred, but make the same change here.)
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.94 3/15 *acm* Fix ticket 2249: attribute handling for user-defined variables
* V698 *acm* 1/16 change remaining dset=-1 to the parameter dset=pdset_uvars

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'	
	include 'rpn.parm'
	include 'xprog_state.cmn'
	include 'xrisc.cmn'
	include 'gfdl_vms.parm'	

* calling argument declaration:
	INTEGER		dset_in, varid, dset_try, status
	CHARACTER*(*)	text, varname, attname

* internal variable declarations:
        LOGICAL coordvar
	INTEGER	TM_LENSTR, TM_LENSTR1, STR_SAME,
     .         dset, tlen, vlen, dot, vartype, nvdims, nvatts, 
     .         vdims(8), all_outflag, first, second, dotdot
        CHARACTER*128 err_string

* Break text at dot: varname.attname

        dset = dset_in
        IF (dset .EQ. unspecified_int4) dset = pdset_uvars   ! try user variables.

        risc_buff = text
        tlen = MIN( size_rbuff, TM_LENSTR(text) )

        IF (tlen .LE. 1 .OR. INDEX(risc_buff,'.') .EQ. 0 ) THEN
           varname = risc_buff
           attname = ""
           GOTO 1000
        ENDIF

        dot = INDEX(risc_buff,'.')
        dotdot = INDEX(risc_buff,'..')
        IF (dotdot .GT. 0) dot = dotdot + 1

* is the name in quotations? If so then it may contain a dot and not
* be varname.attname See if its all within quotes, if so this is a 
* variable name containg a dot.

	first = INDEX(risc_buff(1:tlen), "'" )
	second = INDEX(risc_buff(2:tlen), "'" )
        IF (dot .GT. first .AND. dot .LT. second) THEN
           varname = risc_buff
           attname = ""
           status = ferr_not_attribute
           GOTO 5000
        ENDIF

        tlen = MIN(tlen, LEN(varname))
        tlen = MIN(tlen, LEN(attname))

        varname = risc_buff(:dot-1)
        attname = risc_buff(dot+1:tlen)

* Is the dot a decimal point within an expression, as in LIST a[y=0.5:9.5]
* or LIST a[gx=sst[d=1],y=1.5] Look for opening bracket in varname and
* closing bracket in attname.

        IF ((INDEX(varname,'[') .GT. 0)  .AND. 
     .      (INDEX(attname,']') .GT. 0)) THEN
           varname = risc_buff
           attname = ""
           GOTO 1000
        ENDIF

! varname.attname[d=1]  is ok
! varname[d=1].attname  is not ok

        IF ( INDEX(varname,'[') .GT. 0 ) GOTO 5100 

* Is the varname the keyword . referring to the dataset?
        IF (STR_SAME (varname, '.') .EQ. 0) THEN
           varid = 0
           GOTO 1000
        ENDIF
          
* Check that the dset contains the variable (varid=0 refers to . the 
* pseudovariable that refers to global attributes)

        CALL CD_GET_VAR_ID (dset, varname, varid, status)

* If we didnt find the name in the dataset, it may be a  user var,
* which just depends on that data set

        IF (status  .NE. ferr_ok) THEN
           dset_try = pdset_uvars
           CALL CD_GET_VAR_ID (dset_try, varname, varid, status)
           IF (status .EQ. ferr_ok) dset_in = pdset_uvars
           IF (status .EQ. ferr_ok) dset = pdset_uvars
        ENDIF


        IF (status .NE. ferr_ok) THEN
           status = ferr_unknown_variable
           GOTO 5200
        ENDIF

        vlen = TM_LENSTR1(varname)

        IF (status .NE. ferr_ok .AND. 
     .     (STR_SAME(varname(:vlen), '.') .NE. 0) ) THEN
           status = ferr_unknown_variable
           GOTO 5200
        ENDIF

        CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, nvdims, 
     .                vdims, nvatts, coordvar, all_outflag, status) 

        IF (status .EQ. ferr_ok .AND. 
     .      (coordvar .AND. INDEX( risc_buff(:dot-1), '(' ) .EQ. 0) ) THEN
           status = ferr_syntax
           err_string = "enclose coordinate variables in parentheses"
           GOTO 5300
        ENDIF

        IF ( INDEX(attname,'.') .GT. 0 ) GOTO 5100

* successful completion. 

 1000	status = ferr_ok
	RETURN

* error exits
 5000	CONTINUE
	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, risc_buff(:tlen), *5000 )
 5200	CALL ERRMSG( status, status, risc_buff(:tlen),  *5000 )
 5300	CALL ERRMSG( status, status, risc_buff(:tlen)//pCR//err_string, 
     .               *5000 )

	END
