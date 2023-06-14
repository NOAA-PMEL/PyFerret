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
* V601 11/06*acm* Instead of looking to see if the Ferret axis name and the
*                 axis name in the attribute structure differ because Ferret
*                 needed to rename an axis on file initialization for uniqueness;
*                 we use the Ferret name throughout, restoring it only on output.
* V604 6/07 *acm* Fix bug 1518 If we didnt find the varid in the dataset, it may 
*                 be a user var
* V611 5/08 *acm* Fix bug 1578; dataset specifiers applied correctly var[d=1].attname
* v6.7 1/11 *acm* Fix ticket 1768; do not write nulls when writing err msgs 
*                 with attname.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* v698 5/16  *acm* ticket 2413: New attribute keywords ncoordvars, coordnames
*                  return info about coordinate-variables in datasets

        IMPLICIT NONE
        include 'tmap_dims.parm'
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
     .          recdim, iatt, attoutflag, all_outflag, first, second
        REAL*8 val
        CHARACTER*128 err_string, buff

	CHARACTER*13	TM_STRING, buff13, vbuff
        CHARACTER*1 c

* Break text at dot: varname.attname

        dset = dset_in
        IF (dset .EQ. unspecified_int4) dset = pdset_uvars   ! try user variables.

        tlen = TM_LENSTR(text)

        tlen = MIN(tlen, LEN(varname))
        tlen = MIN(tlen, LEN(attname))

        IF (tlen .LT. 1 .OR. INDEX(text,'.') .EQ. 0 ) THEN
           varname = text
           attname = ""
           status = ferr_not_attribute
           GOTO 1000
        ENDIF

        dot = INDEX(text,'.')
        dotdot = INDEX(text,'..')
        IF (dotdot .GT. 0) dot = dotdot + 1

        IF (dot .EQ. 1  .AND.  tlen .EQ. 1) THEN
           varname = text
           attname = ""
           status = ferr_not_attribute
           GOTO 1000
        ENDIF

* is the name in quotations? If so then it may contain a dot and not
* be varname.attname See if its all within quotes, if so this is a 
* variable name containg a dot.

	first = INDEX(text(1:tlen), "'" )
	second = INDEX(text(2:tlen), "'" )
        IF (dot .GT. first .AND. dot .LT. second) THEN
           varname = text
           attname = ""
           status = ferr_not_attribute
           GOTO 5000
        ENDIF

        varname = text(:dot-1)
        attname = text(dot+1:tlen)

* Is the dot a decimal point within an expression, as in LIST a[y=0.5:9.5]
* or LIST a[gx=sst[d=1],y=1.5] Look for opening bracket in varname and
* closing bracket in attname.

        IF ((INDEX(varname,'[') .GT. 0)  .AND. 
     .      (INDEX(attname,']') .GT. 0)) THEN
           varname = text
           attname = ""
           GOTO 1000
        ENDIF

* If attname has brackets on it, remove this part for the next tests

       alen = TM_LENSTR1(attname)
       end_att = alen
       IF (INDEX(attname,'[') .GT. 0) THEN
          end_att = INDEX(attname,'[')
          attname(end_att:alen) = ' '
       ENDIF

* Is the varname the keyword ., representing the dataset? Check that the 
* attname request is valid.

       IF (STR_SAME (varname, '.') .EQ. 0) THEN
          attok = .FALSE.
          IF (STR_SAME(attname, 'varnames') .EQ. 0 ) THEN
             attok = .TRUE.
          ELSE IF (STR_SAME(attname, 'dimnames') .EQ. 0 ) THEN
             attok = .TRUE.
          ELSE IF (STR_SAME(attname, 'coordnames') .EQ. 0 ) THEN
             attok = .TRUE.
          ELSE IF (STR_SAME(attname, 'attnames') .EQ. 0 ) THEN
             attok = .TRUE.
          ELSE IF (STR_SAME(attname, 'ndims') .EQ. 0 ) THEN
             attok = .TRUE.
          ELSE IF (STR_SAME(attname, 'nattrs') .EQ. 0 ) THEN
             attok = .TRUE.
          ELSE IF (STR_SAME(attname, 'nvars') .EQ. 0 ) THEN
             attok = .TRUE. 
          ELSE IF (STR_SAME(attname, 'ncoordvars') .EQ. 0 ) THEN
             attok = .TRUE. 

* Is the reference to a global attribute?

          ELSE
             CALL CD_GET_VAR_ID (dset, '.', varid, status)
             CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts, 
     .               recdim, status)

             DO 100 iatt = 1, ngatts
                CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, buff, 
     .                    status)
                IF (STR_SAME(buff, attname) .EQ. 0) attok = .TRUE.
 100         CONTINUE
          ENDIF

          IF (attok) THEN
             varid = 0
             GOTO 1000
          ELSE

* is the attribute name in quotations? If so then test without them now: When the
* above keywords, attnames, etc, are in the file as attribute names, then 
* they must be enclosed in quotes

             alen = TM_LENSTR1(attname)
             IF  (attname(1:1) .EQ. "'" .AND. 
     .            attname(alen:alen) .EQ. "'") THEN

                CALL CD_GET_VAR_ATT_ID (dset, varid, attname, 
     .              attid, status)
                IF (status .EQ. ferr_ok) GOTO 1000
             ENDIF
	     
             CALL CD_GET_VAR_ATT_ID (dset, varid, attname, 
     .              attid, status)
             IF (status .EQ. ferr_ok) GOTO 1000
 
             attok = .FALSE.
             IF (status .EQ. atom_not_found) status = 
     .               ferr_unknown_attribute
             vlen = TM_LENSTR1(varname)
             err_string = 'attribute '//attname(:alen)//
     .                       ' not found for dataset '
             GOTO 5200

          ENDIF

       ENDIF
          
* Check that the dset contains the variable; . is varid=0
        
* Is the variable name in quotations?
        
        vlen = TM_LENSTR1(varname)
        IF (varname(1:1) .EQ. "'" .AND. varname(vlen:vlen) .EQ. "'") THEN
           status = 3  ! just for debugging
        ENDIF 

        CALL CD_GET_VAR_ID (dset, varname, varid, status)

* If we didnt find the name in the dataset, it may be a  user var,
* which just depends on that data set

        IF (status  .NE. ferr_ok) THEN
           dset = pdset_uvars
           CALL CD_GET_VAR_ID (dset, varname, varid, status)
           IF (status .EQ. ferr_ok) dset = pdset_uvars
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
        varname = buff  ! get varname in original upper- or lowercase form

        IF ( INDEX(attname,'.') .GT. 0 ) THEN
           GOTO 5100
        ENDIF

       IF (coordvar .AND. INDEX( text(:dot-1), '(' ) .EQ. 0 ) THEN
          status = ferr_syntax
          err_string = "enclose coordinate variable name in parentheses"
          GOTO 5300
       ENDIF

       err_string = 'attribute '//attname(:alen)//
     .              ' not found for variable '// varname(:vlen)

* Is the attribute specified by a number? (check as in find_dset_number)
	IF ( TM_DIGIT( attname ) ) THEN

	   buff = attname(:alen)
	   READ (buff,*, ERR=5200 ) attid
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

           CALL CD_GET_VAR_ATT_ID(dset, varid, attname, attid, status)
           IF (status .NE. ferr_ok .OR. attid .EQ. 0) THEN

* See if it is a pseudo-attribute attnames, ndims, etc.

              IF (STR_SAME(attname, 'attnames') .EQ. 0 ) THEN 
                 attok = .TRUE.
              ELSE IF (STR_SAME(attname, 'dimnames') .EQ. 0 ) THEN
                 attok = .TRUE.
              ELSE IF (STR_SAME(attname, 'ndims') .EQ. 0 ) THEN
                 attok = .TRUE.
              ELSE IF (STR_SAME(attname, 'nctype') .EQ. 0 ) THEN
                 attok = .TRUE.
              ELSE IF (STR_SAME(attname, 'nattrs') .EQ. 0 ) THEN
                 attok = .TRUE.
              ELSE

* is the attribute name in quotations? If so then remove them, and test
* for the name with case-sensitivity. When the above keywords, attnames, 
* etc, are in the file as attribute names, then they must be enclosed in 
* quotes to be accessed.

                 alen = TM_LENSTR1(attname)
                 IF  (attname(1:1) .EQ. "'" .AND. 
     .                attname(alen:alen) .EQ. "'") THEN
                    
                    CALL CD_GET_VAR_ATT_ID (dset, varid, attname,
     .                 attid, status ) 
                    IF (status .EQ. ferr_ok) GOTO 1000
                 ENDIF
 
                 attok = .FALSE.
                 IF (status .EQ. atom_not_found) status = 
     .               ferr_unknown_attribute
                 vlen = TM_LENSTR1(varname)
                 err_string = 'attribute '//attname(:alen)//
     .                       ' not found for variable '//
     .                        varname(:vlen)
                 GOTO 5200
              ENDIF
           ENDIF

       ENDIF

* successful completion
        dset_in = dset
 1000	status = ferr_ok

	RETURN

* error exits. ADD_ATTRIBUTE uses this routine to see if an attribute is already
* defined, and if not, creates the attribute.  In that case we do not want the 
* error msg.

 5000	CONTINUE
	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, text(:tlen), *5000 )
        RETURN

 5200	IF (do_err) THEN
           buff = text(:tlen)
           CALL ERRMSG( status, status, buff(:tlen)//pCR//err_string, 
     .               *5000 )
        ELSE
           RETURN
        ENDIF

 5300	CALL ERRMSG( status, status, err_string, *5000 )
        RETURN

	END
