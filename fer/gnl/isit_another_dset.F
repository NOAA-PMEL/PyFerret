      SUBROUTINE ISIT_ANOTHER_DSET (expression, dset, varname, status)

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
* Look for [d=n] within an expression, and reset dset to match the dataset number
* or name. varname is returned without the [d=] expression.

* V600 6/05 *acm* Attribute control.

	INCLUDE	'ferret.parm'
	INCLUDE 'errmsg.parm'

* calling argument declarations

      INTEGER dset, status
      CHARACTER*(*) expression, varname

* local variable declarations
      INTEGER TM_LENSTR1, slen, elen, llen, ivar, mods_cx, category

      llen = TM_LENSTR1(expression)
      varname = expression

      IF ( INDEX(expression,'[') .GT. 0 ) THEN

         dset = pdset_irrelevant	! default (not used ...)
         CALL PARSE_NAM_DSET(expression, cx_last, dset, category, 
     .                       ivar, mods_cx, status)
         IF (status .NE. ferr_ok) GOTO 5010
         slen = INDEX(expression,'[') 
         elen = INDEX(expression,']')
         varname = expression(1:slen-1)
         IF (llen .GT. elen) 
     .      varname = expression(1:slen-1)//expression(elen+1:llen)
 
      ENDIF

5000  RETURN

5010  IF (dset .EQ. pdset_irrelevant .OR. 
     .       dset .EQ. unspecified_int4) status = ferr_syntax

      RETURN
      END
