	SUBROUTINE EQUAL_FORMAT( string, status )

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
* from a string of the form "FORMAT=xxxx" determine the listing format and
* save the information in COMMON

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:   9/1/89
*	 10/1/90 - "BIBO" --> "EPIC"
* V230:  6/92 - use common variable list_fmt_type
*       10/92 - improve test for I format by checking for digit following "i"
*       11/92 - get lower case format
* V301:  1/94 - extracted code into CHECK_FORMAT
* V320: 12/94 - added /FORMAT=STREAM
* V420:  9/95 - added /FORMAT=COMMA-DELIMITED and /FORMAT=TAB-DELIMITED
* V450: 11/96 - added /FORMAT=CACHE for DODS caching

* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string

* internal variable declarations
	INTEGER	STR_UPCASE, vax_code
	CHARACTER buff*80, upcase3*3

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* SET LIST/FORMAT=(FORTRAN_FORMAT),'UNFORMATTED','EPIC','TMAP','GT','STREAM'

* remove old specs
	list_format_given = .FALSE.

        CALL EQUAL_STR_LC( string, buff, status )   ! lower case 11/92
        IF ( status .NE. ferr_ok ) RETURN
	IF ( buff .NE. ' ' ) list_format = buff

* special interpretations
        vax_code = STR_UPCASE( upcase3, list_format(1:3) )
        IF (     upcase3 .EQ. 'UNF' ) THEN
           list_fmt_type = plist_unformatted
        ELSEIF ( upcase3 .EQ. 'EPI' ) THEN
           list_fmt_type = plist_epic
        ELSEIF ( upcase3 .EQ. 'TMA' ) THEN
           list_fmt_type = plist_gt
        ELSEIF ( upcase3 .EQ. 'GT' ) THEN
           list_fmt_type = plist_gt
        ELSEIF ( upcase3 .EQ. 'CDF' ) THEN
           list_fmt_type = plist_cdf
        ELSEIF ( upcase3 .EQ. 'STR' ) THEN
           list_fmt_type = plist_stream
        ELSEIF ( upcase3 .EQ. 'COM' ) THEN
           list_fmt_type = plist_comma_del
        ELSEIF ( upcase3 .EQ. 'TAB' ) THEN
           list_fmt_type = plist_tab_del
        ELSEIF ( upcase3 .EQ. 'CAC' ) THEN
           list_fmt_type = plist_cdf_cache
        ELSE

* ... check out user-specified format
	   CALL CHECK_FORMAT( list_format, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   list_format_given = .TRUE.
           list_fmt_type = plist_formatted
	ENDIF

* success
	status = ferr_ok
	RETURN

* syntax error
! 5000	RETURN
	END
