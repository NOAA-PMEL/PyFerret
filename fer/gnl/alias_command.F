	SUBROUTINE ALIAS_COMMAND( buffer, clen, * )

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
* check to see if an alias was used to abbreviate the command and replace with
* the full string if so

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  2/8/90
* V230:  1/27/92 - added QUIT, DATA, and REGION
*           2/92 - improved logic to permit aliases of less than 4 chars
*         6/8/92 - upper bound of aliases changed to reflext DEFINE ALIAS
*        6/19/92 - alias names 8 characters
*        7/29/92 - uses ALIAS_ID
* V300: take abnormal return if alias substitution occurs
* V313: *sh* AIX port.  logic for replacing buffer doesn't work
* v4.5: 8/97 *kob* AIX bug fix - added check for cpos to locate tabs in
*                           command line
*     :10/97 *kob* Check to see if alen = clen. if so set buffer to the 
*                  alias.  new version of DEC fortran caught the error in
*                  the string concatenation.
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead
* *sh* 11/2013 - allow an "=" as the end of an alias, soo "vtree=all" becomes legal
 
        IMPLICIT NONE
* calling argument declarations:
	INTEGER	  clen
	CHARACTER buffer*(*)

* internal variable declarations:
	INTEGER	TM_LENSTR1, ALIAS_ID, 
     .          alen, true_len, i, spos, bpos, cpos, epos

        CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

	include 'ferret.parm'
	include 'xcommand.cmn'

* what is the alias command name length - terminated by "/" or " " (or tab or "=")
        alen = 8
        spos = INDEX(buffer(:alen),'/')
        bpos = INDEX(buffer(:alen),' ') 
* 8/97 check for tab - it can be end of alias
	cpos = INDEX(buffer(:alen),tab)
	IF ( cpos .EQ. 0 ) cpos = clen + 1
	IF ( bpos .EQ. 0 ) bpos = clen + 1
	bpos = MIN( cpos, bpos )
* 11/13 similar check for "="
	epos = INDEX(buffer(:alen),'=')
	IF ( epos .EQ. 0 ) epos = clen + 1
	bpos = MIN( epos, bpos )
        IF ( spos .NE. 0 ) alen = spos - 1
        alen = MIN( alen, bpos - 1 )
        
* identify the alias
        i = ALIAS_ID( buffer(:alen) ) 

* substitute alias text if appropriate
        IF ( i .NE. unspecified_int4 ) THEN
	   true_len = TM_LENSTR1( alias(i) )
* 10/97 *kob* added check to see if passed in length is same as alias lenght
	   IF (alen .EQ. clen) THEN
	      buffer = alias(i)
	   ELSE
	      buffer = alias(i)(:true_len)//buffer(alen+1:clen)
	   ENDIF
	   clen = clen + ( true_len - alen )	   
	   RETURN 1    ! abnormal return if substitution
	ENDIF

* no substitution
	RETURN
	END
