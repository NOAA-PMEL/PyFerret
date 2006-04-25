	SUBROUTINE BREAK_VARATTNAME( text, dset_in, varname, attname, 
     .                               varid, do_err, status )
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
* Check whether the variable and attribute exist within the dataset
* stored in linked-list structure of nc datasets.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V600 6/05 *acm* Attribute control.

	include	'ferret.parm'
	include 'errmsg.parm'	
	include 'rpn.parm'
	include 'xprog_state.cmn'
	include 'xrisc.cmn'
	include 'gfdl_vms.parm'	

* calling argument declaration:
        LOGICAL         do_err
	INTEGER		dset_in, status
	CHARACTER*(*)	text, varname, attname

* internal variable declarations:
        LOGICAL TM_DIGIT, coordvar, attok
	INTEGER	TM_LENSTR, TM_LENSTR1, STR_SAME, STR_CASE_BLIND_COMPARE,
     .          dset, tlen, alen, vlen, dot, dotdot, varid, attlen, 
     .          attype, end_att, short_len, attid, vartype, nvdims, 
     .          nvatts, vdims(8), vax_code, ndims, nvars, ngatts, 
     .          recdim, iatt, attoutflag, all_outflag, first, second, ic
        REAL*8 val
        CHARACTER*128 err_string, buff

	CHARACTER*13	TM_STRING, buff13, vbuff
        CHARACTER*1 c

* Break text at dot: varname.attname

        dset = dset_in
        IF (dset .EQ. unspecified_int4) dset = -1   ! try user variables.

        risc_buff = text
        tlen = MIN( size_rbuff, TM_LENSTR(text) )

        tlen = MIN(tlen, LEN(varname))
        tlen = MIN(tlen, LEN(attname))

        IF (tlen .LT. 1 .OR. INDEX(risc_buff,'.') .EQ. 0 ) THEN
           varname = risc_buff
           attname = ""
           status = ferr_not_attribute
           GOTO 1000
        ENDIF

        dot = INDEX(risc_buff,'.')
        dotdot = INDEX(risc_buff,'..')
        IF (dotdot .GT. 0) dot = dotdot + 1

        IF (dot .EQ. 1  .AND.  tlen .EQ. 1) THEN
           varname = risc_buff
           attname = ""
           status = ferr_not_attribute
           GOTO 1000
        ENDIF

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

* Is the varname the keyword ., representing the dataset? Check that the 
* attname request is valid.

        IF (STR_SAME (varname, '.') .EQ. 0) THEN
           attok = .FALSE.
           IF (STR_SAME(attname(:4), 'varn') .EQ. 0 ) THEN
              attok = .TRUE.
              attname = 'varnames'
           ELSE IF (STR_SAME(attname(:4), 'dimn') .EQ. 0 ) THEN
              attok = .TRUE.
              attname = 'dimnames'
           ELSE IF (STR_SAME(attname(:3), 'att') .EQ. 0 ) THEN
              attok = .TRUE.
              attname = 'attnames'
           ELSE IF (STR_SAME(attname(:4), 'ndim') .EQ. 0 ) THEN
              attok = .TRUE.
              attname = 'ndims'
           ELSE IF (STR_SAME(attname(:3), 'nat') .EQ. 0 ) THEN
              attok = .TRUE.
              attname = 'natts'
           ELSE IF (STR_SAME(attname(:4), 'nvar') .EQ. 0 ) THEN
              attok = .TRUE.
              attname = 'nvars'

* Is the reference to a global attribute?

           ELSE
              CALL CD_GET_VAR_ID (dset, '.', varid, status)
              CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts, 
     .               recdim, status)

              DO 100 iatt = 1, ngatts
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, buff, 
     .                    status)
                 IF (STR_SAME(buff, attname) .EQ. 0) attok = .TRUE.
 100          CONTINUE
           ENDIF

           IF (attok) THEN
              varid = 0
              GOTO 1000
           ELSE
              varid = -1
              status = ferr_unknown_attribute
              err_string = "invalid dataset attribute requested"
              GOTO 5200
           ENDIF
        ENDIF
          
* Check that the dset contains the variable; . is varid=0
        
        CALL CD_GET_VAR_ID (dset, varname, varid, status)
        vlen = TM_LENSTR1(varname)

* If this is a coord var and the name matched an existing different axis 
* name on input, then Ferret appends a digit to the name. 

        buff = varname

        IF (status .NE. ferr_ok .AND. 
     .      varname(1:1) .EQ. '('  .AND.
     .      varname(vlen:vlen) .EQ. ')' ) THEN

* test that only extra digits were added
            DO 80 ic = vlen, 1, -1
	       c = varname(ic:ic)
	       IF (  c .LT. '0' .OR. c .GT. '9' ) GOTO 90

* axis names must be enclosed in parentheses to get a varid returned from cd_get_var_id
               buff = varname(1:ic-1)//')'
               CALL CD_GET_VAR_ID (dset, buff, varid, status)

               IF (varid .GT. 0) GOTO 90

 80	    CONTINUE
        ENDIF

        IF (status .NE. ferr_ok .AND. 
     .      (STR_SAME(varname(:vlen), '.') .NE. 0) ) THEN
           status = ferr_unknown_variable
           err_string = " "
           GOTO 5200
        ENDIF

  90    CONTINUE
        CALL CD_GET_VAR_INFO (dset, varid, buff, vartype, nvdims, 
     .                vdims, nvatts, coordvar, all_outflag, status) 

        IF ( INDEX(attname,'.') .GT. 0 ) THEN
           GOTO 5100
        ENDIF

        IF (coordvar .AND. INDEX( risc_buff(:dot-1), '(' ) .EQ. 0 ) THEN
           status = ferr_syntax
           err_string = "enclose coordinate variable name in parentheses"
           GOTO 5300
        ENDIF

* If attname has brackets on it, remove this part for the next tests

       alen = TM_LENSTR1(attname)
       end_att = alen
       IF (INDEX(attname,'[') .GT. 0) THEN
          end_att = INDEX(attname,'[')
          attname(end_att:alen) = ' '
       ENDIF

       err_string = 'attribute '//attname//
     .              ' not found for variable '// varname(:vlen)

* Is the attribute specified by a number? (check as in find_dset_number)
	IF ( TM_DIGIT( attname ) ) THEN

	   buff = attname

#ifdef MANDATORY_FORMAT_WIDTHS
	   READ (buff,*, ERR=5200 ) attid
#else
	   READ (buff, '(I)', ERR=5200 ) attid
#endif
           val = attid
           buff13 = TM_STRING(val)
           CALL TM_PACK_STRING(buff13, vbuff, 1, 13, short_len)
	   vax_code = STR_CASE_BLIND_COMPARE( vbuff, buff(1:13) )
	   IF (  vax_code .NE. vms_str_success ) GOTO 5200

* Check that this var has this attribute.  
           IF (attid .LT. 1   .OR.  attid .GT. nvatts) GOTO 5200
           CALL CD_GET_VAR_ATT_NAME (dset, varid, attid, attname, 
     .              status)
	ELSE

           CALL CD_GET_VAR_ATT_INFO(dset, varid, attname,
     .        attype, attlen, attoutflag, status )
           IF (status .NE. ferr_ok) THEN

* See if it is a pseudo-attribute attnames, ndims, etc.

             IF (STR_SAME(attname(:4), 'attn') .EQ. 0 ) THEN  ! attnames
                attname = 'attnames'
             ELSE IF (STR_SAME(attname(:4), 'dimn') .EQ. 0 ) THEN  ! dimnames
                attname = 'dimnames'
             ELSE IF (STR_SAME(attname(:4), 'ndim') .EQ. 0 ) THEN  ! ndims
                attname = 'ndims'
             ELSE IF (STR_SAME(attname(:4), 'type') .EQ. 0 ) THEN
                attname = 'type'
              ELSE IF (STR_SAME(attname(:4), 'natt') .EQ. 0 ) THEN  ! natts
                attname = 'natts'
             ELSE
                IF (status .EQ. atom_not_found) status = 
     .               ferr_unknown_attribute
                vlen = TM_LENSTR1(varname)
                err_string = 'attribute '//attname//
     .                       ' not found for variable '//
     .                        varname(:vlen)
                GOTO 5200
             ENDIF
          ENDIF

       ENDIF

* successful completion
 1000	status = ferr_ok

	RETURN

* error exits. ADD_ATTRIBUTE uses this routine to see if an attribute is already
* defined, and if not, creates the attribute.  In that case we do not want the 
* error msg.

 5000	CONTINUE
	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, risc_buff(:tlen), *5000 )
        RETURN

 5200	IF (do_err) THEN
           CALL ERRMSG( status, status, risc_buff(:tlen)//pCR//err_string, 
     .               *5000 )
        ELSE
           RETURN
        ENDIF

 5300	CALL ERRMSG( status, status, err_string, *5000 )
        RETURN

	END
