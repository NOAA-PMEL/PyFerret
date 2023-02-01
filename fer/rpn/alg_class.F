	INTEGER FUNCTION ALG_CLASS( ctr )

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
* subroutine for lexical analyzer
* determine chatacter class of character
* note: the routine could be made simpler and faster by simply providing a
*	DATA statement associating each character number (ICHAR('C')) with a
*	class.   ... hurried development ...

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V200: 11/24/89
* V495: 7/7/98 - add single quotes as another character class (11) (old 11->12)
*     : 9/30/98 - NAG F90 (linux) received error on above change.  
*                 need to put single  tick between double quotes  *kob*
* V500: 12/98 - add "{" and "}" to support "{1,3,5}" notation
*	 1/99 - allow strings in quoble quotes (as fcn args)
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* 12/14 *sh* "_" is a separate character class needed for _DQ_ and _SQ_

        IMPLICIT NONE
	include	'ferret.parm'

* calling argument declaration:
	CHARACTER*1	ctr

* internal parameter declarations:
	CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

* classes of characters
*	C1 - + -
*	C2 - digits
*	C3 - letters
*	C4 - [
*	C5 - ]
*	C6 - * / ^ , (
*	C7 - )
*	C8 - blank tab
*	C9 - E e
*	C10- .
*       C11- '
*	C12- {
*	C13- }
*	C14- "
*	C15- _
*	C15- all others

	IF     ( ctr.EQ.'+' .OR.  ctr.EQ.'-' ) THEN
	   ALG_CLASS = 1
	ELSEIF ( ctr.GE.'0' .AND. ctr.LE.'9' ) THEN
	   ALG_CLASS = 2
	ELSEIF ( ctr.EQ.'E' .OR.  ctr.EQ.'e' ) THEN
	   ALG_CLASS = 9
	ELSEIF ( ( ctr.GE.'A' .AND. ctr.LE.'Z' )
     .	    .OR. ( ctr.GE.'a' .AND. ctr.LE.'z' ) ) THEN
	   ALG_CLASS = 3
	ELSEIF ( ctr.EQ.'[' ) THEN
	   ALG_CLASS = 4
	ELSEIF ( ctr.EQ.']' ) THEN
	   ALG_CLASS = 5
	ELSEIF ( ctr.EQ.'*' .OR. ctr.EQ.'/' .OR. ctr.EQ.'^'
     .	    .OR. ctr.EQ.',' .OR. ctr.EQ.'('		) THEN
	   ALG_CLASS = 6
	ELSEIF ( ctr.EQ.')' ) THEN
	   ALG_CLASS = 7
	ELSEIF ( ctr.EQ.' ' .OR. ctr.EQ.tab ) THEN
	   ALG_CLASS = 8
	ELSEIF ( ctr.EQ.'.' ) THEN
	   ALG_CLASS = 10
	ELSEIF ( ctr.EQ. "'" ) THEN
	   ALG_CLASS = 11
	ELSEIF ( ctr.EQ.'{' ) THEN
	   ALG_CLASS = 12
	ELSEIF ( ctr.EQ.'}' ) THEN
	   ALG_CLASS = 13
	ELSEIF ( ctr.EQ.'"' ) THEN
	   ALG_CLASS = 14
	ELSEIF ( ctr.EQ.'_' ) THEN
	   ALG_CLASS = 15
	ELSE
	   ALG_CLASS = 16
	ENDIF

	RETURN
	END
