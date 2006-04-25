      SUBROUTINE ISIT_COORD_VAR ( expression, dset, varname, 
     .                            coordvar, status )

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
      
* See if the expression is a coordinate variable name from the linked-list structure, 
* inside parentheses.  Return the dset and logical flag coordvar.  If the expression
* is not in parentheses, it is not treated as a variable name.

* V600 6/05 *acm* Attribute control.

        INCLUDE	'ferret.parm'
	INCLUDE 'errmsg.parm'
        INCLUDE 'xrisc.cmn'


* calling argument declarations
      LOGICAL coordvar
      INTEGER dset, status
      CHARACTER*(*) expression, varname

* local variable declarations

      INTEGER TM_LENSTR1, NCF_GET_VAR_AXFLAG, slen, varid, ivar, 
     .        mods_cx, category, itmp, dir, ic
      CHARACTER*128 buff
      CHARACTER*1   c
   
      coordvar = .FALSE.

      slen = TM_LENSTR1(expression)
      buff = expression

      IF (expression(1:1) .EQ. '(' .AND.
     .    INDEX(expression(1:slen), ')' ) .GT. 1 ) THEN

         varname = expression(2:slen)

* locate the end of the parentheses

         itmp = INDEX(varname,')')
         IF ( itmp .GT. slen .OR. itmp .LE. 0) GOTO 5200   ! unpaired parentheses
         varname(itmp:itmp) = ' '
         itmp = itmp - 1
         slen = itmp

         IF ( INDEX(varname,'[') .GT. 0 ) THEN

            dset = pdset_irrelevant	! default (not used ...)
            CALL PARSE_NAM_DSET(varname, cx_last, dset,
     .        category, ivar, mods_cx, status)
            IF (status .NE. ferr_ok) GOTO 5000
            slen = INDEX(varname,'[') 
            varname(slen:) = ' ' 
         ENDIF
         IF (dset .EQ. pdset_irrelevant .OR. 
     .       dset .EQ. unspecified_int4) GO TO 5100

         CALL CD_GET_VAR_ID (dset, varname(:slen), varid, status)
     
         IF (status .NE. ferr_ok) THEN

* The name may be axis name with digit(s) added for a unique name. 
* test that only extra digits were added

            DO 80 ic = slen, 1, -1
	       c = varname(ic:ic)
	       IF (  c .LT. '0' .OR. c .GT. '9' ) GOTO 90

* axis names must be enclosed in parentheses to get a varid returned from cd_get_var_id
c               buff = '('//varname(1:ic-1)//')'
               buff = varname(1:ic-1)
               CALL CD_GET_VAR_ID (dset, buff, varid, status)

               IF (varid .GT. 0) THEN
* Remove the digit 
                  buff(ic:llen) = ' '
                  GOTO 90

               ENDIF

 80	    CONTINUE
         ENDIF

 90      CONTINUE      

         IF (status .NE. ferr_ok) GOTO 5100

         status = NCF_GET_VAR_AXFLAG (dset, varid, coordvar, dir)

      ELSE
 
         varname = expression

         IF ( INDEX(varname,'[') .GT. 0 ) THEN

            dset = pdset_irrelevant	! default (not used ...)
            CALL PARSE_NAM_DSET(varname, cx_last, dset,
     .        category, ivar, mods_cx, status)
            IF (status .NE. ferr_ok) GOTO 5000
            slen = INDEX(varname,'[') 
            varname(slen:) = ' ' 
            slen = slen - 1
         ENDIF
         IF (dset .EQ. pdset_irrelevant .OR. 
     .       dset .EQ. unspecified_int4) GO TO 5100

         CALL CD_GET_VAR_ID (dset, varname(:slen), varid, status)
      
         IF (status .NE. ferr_ok) GOTO 5100

         status = NCF_GET_VAR_AXFLAG (dset, varid, coordvar, dir)  

      ENDIF

* error exits
 5000 RETURN
 5100 status = ferr_invalid_command
      RETURN
 5200 status = ferr_syntax
      CALL ERRMSG( status, status, 'no closing parentheses'//
     .			buff(:slen), *5000 )
      END
