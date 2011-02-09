      SUBROUTINE INHERIT_ATTRS (varname1, varname2, status)
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

* Get all attributes from varname1, a variable in dataset dset1, and assign to the 
* variable in varname2, dataset dset2.  dset2 has been determined in xeq_set.F
* V600  4/06 *acm*   attribute handling
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* V670  2/11 *acm*   Fixes for inheriting attributes.

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'      
      INCLUDE 'xprog_state.cmn'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'

* Argument declarations
      CHARACTER varname1*(*), varname2*(*)
      INTEGER status

* Local variable declarations
      LOGICAL coordvar
      INTEGER TM_LENSTR, slen, tlen, dset1, dset2, 
     .  varid1, varid2, vtype, nvdims, nvatts1, nvatts2,
     .  vdims(8), iatt, first_quote, next_quote, dot, mr, cx, 
     .  all_outflag, cat
      CHARACTER*128 attname, buff1, buff2

* Check that neither buffer contains varname.attname (If it is in 
* quotations then the dot is part of the variable name.)

      tlen = TM_LENSTR(varname1) 
      first_quote = INDEX(varname1(1:tlen), "'" )
      next_quote = INDEX(varname1(2:tlen), "'" )

      IF (INDEX(varname1,'.') .NE. 0 ) THEN
         dot = INDEX(varname1,'.')
         IF (dot .GT. first_quote .AND. dot .LT. next_quote) GOTO 5010
      ENDIF

* and now for varname2
      tlen = TM_LENSTR(varname2) 

      first_quote = INDEX(varname2(1:tlen), "'" )
      next_quote = INDEX(varname2(2:tlen), "'" )

      IF (INDEX(varname2,'.') .NE. 0 ) THEN
         dot = INDEX(varname2,'.')
         IF (dot .GT. first_quote .AND. dot .LT. next_quote) GOTO 5010
      ENDIF

* Do the varnames contain dataset specifications?
* evaluate the context without computing the expression to get the dataset

      CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
      IF ( status .NE. ferr_ok ) GOTO 5000
 
      CALL EVAL_CONTEXT ( cx_cmnd, varname1, status )
      IF ( status .NE. ferr_ok ) GOTO 5000

      mr = is_mr( 1 )
      cx = is_cx( 1 )

      dset1 = cx_data_set(cx) 
      IF (dset1 .EQ. pdset_irrelevant) dset1 = cx_data_set(cx_last) 

      cat = cx_category(cx)
      IF (cat .EQ. cat_user_var) dset1 = -1

      CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
      IF ( status .NE. ferr_ok ) GOTO 5000
 
      CALL EVAL_CONTEXT ( cx_cmnd, varname2, status )
      IF ( status .NE. ferr_ok ) GOTO 5000

      mr = is_mr( 1 )
      cx = is_cx( 1 )

      dset2 = cx_data_set(cx) 
      IF (dset2 .EQ. pdset_irrelevant) dset2 = cx_data_set(cx_last) 

      cat = cx_category(cx)
      IF (cat .EQ. cat_user_var) dset2 = -1

* Get varid for each variable.
      CALL CD_GET_VAR_ID (dset1, varname1, varid1, status)
      IF (status .NE. ferr_ok) GOTO 5050
      CALL CD_GET_VAR_ID (dset2, varname2, varid2, status)
      IF (status .NE. ferr_ok) GOTO 5060

* Get number of attributes for variables

      CALL CD_GET_VAR_INFO (dset1, varid1, varname1, vtype, nvdims, vdims,
     .       nvatts1, coordvar, all_outflag, status)
      IF (coordvar) GOTO 5020

      CALL CD_GET_VAR_INFO (dset2, varid2, varname2, vtype, nvdims, vdims,
     .       nvatts2, coordvar, all_outflag, status)
      IF (coordvar) GOTO 5020

* Remove attributes from variable 2

      DO 100 iatt = 1, nvatts2
         CALL CD_GET_VAR_ATT_NAME (dset2, varid2, iatt, attname, status)
         CALL CD_DELETE_ATTRIBUTE (dset2, varid2, attname, status)
         IF (status .NE. ferr_ok) GOTO 5030
 100  CONTINUE

* Get attribute information from variable 1 and put onto variable 2.
* Do this in the c library so we dont have to define a fixed string for
* string type attributes.

      iflag = 1
      DO 200 iatt = 1, nvatts1
         CALL CD_TRANSFER_ATTRIBUTE (dset1, varid1, iatt, 
     .        dset2, varid2, status)
         IF (status .NE. ferr_ok) GOTO 5040
         CALL CD_GET_VAR_ATT_NAME (dset1, varid1, iatt, attname, status)
	 CALL CD_SET_ATT_FLAG (dset2, varid2, attname, iflag, status)
 200  CONTINUE

* Update attribute information used by plotting etc (ds_units, ...)

        CALL UPDATE_ATTRIBUTES (dset2, varid2)

 5000 RETURN

 5010 CALL ERRMSG( ferr_invalid_command, status,
     .  'Use SET ATT/LIKE=var2 var1 without attribute names', *5000 )

 5020 CALL ERRMSG (ferr_invalid_command, status,
     . 'Cannot inherit attributes to/from coordinate variables', *5000)

 5030 buff2 = varname2
      CALL ERRMSG (ferr_internal, status,
     . 'Error removing attributes from '//buff2(:tlen), *5000)

 5040  slen = TM_LENSTR(varname1)
       buff1 = varname1
       CALL ERRMSG (ferr_internal, status,
     . 'Error transferring attributes from '//buff1(:slen)//' to '//
     .  buff2(:tlen), *5000)

 5050  slen = TM_LENSTR(varname1)
       buff1 = varname1
       CALL ERRMSG (ferr_var_not_in_set, status,
     . varname1(:slen),*5000)

 5060  slen = TM_LENSTR(varname1)
       buff1 = varname1
       CALL ERRMSG (ferr_var_not_in_set, status,
     . varname1(:slen),*5000)

      END
