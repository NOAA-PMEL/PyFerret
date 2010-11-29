      SUBROUTINE GET_INPUT_VARNAME (varattname, varname)

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
      
* Return the varname part of an varattname, without any parentheses or brackets

* V600 6/05 *acm* Attribute control.
* V600  4/06 *acm* - additional declarations found by Solaris compiler

        INCLUDE 'xrisc.cmn'
        INCLUDE 'errmsg.parm'

* calling argument declarations

      CHARACTER*(*) varattname, varname

* local variable declarations

      INTEGER TM_LENSTR1, slen, itmp, status

      slen = TM_LENSTR1(varattname)
      IF (varattname(1:1) .EQ. '(' .AND.
     .    INDEX(varattname(1:slen), ')' ) .GT. 1 ) THEN

         varname = varattname(2:slen)

* locate the end of the parentheses

         itmp = INDEX(varname,')')
         IF ( itmp .GT. slen ) GOTO 5200		! unpaired parentheses
         varname(itmp:slen) = ' '

* Remove square brackets
         IF ( INDEX(varname,'[') .GT. 0 ) THEN
            slen = INDEX(varname,'[') 
            varname(slen:) = ' ' 
         ENDIF
      ENDIF

* error exits
 5000 RETURN
 5200 status = ferr_syntax
      risc_buff = varattname
      CALL ERRMSG( status, status, 'no closing parentheses'//
     .	           risc_buff(:slen), *5000 )
      END
