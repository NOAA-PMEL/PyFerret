	SUBROUTINE WRITE_MULTI( lun, string )

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
* write multiple lines of the form line//CR//line... to unit lun

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200: 11/27/89
* 	 2/20/90 - corrected string length bug for more-than 2 lines
* V300: adapted for output to GUI displa
* V314: 8/22/94 *kob* IBM port - need to use risc buff for char. concat 
* V320: 2/28/95 - SPLIT_GUI_MESSAGE --> TM_SPLIT_MESSAGE
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface
*                TM_SPLIT_MESSAGE replaced by SPLIT_LIST

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
#ifdef NO_PASSED_CONCAT
	include 'xrisc.cmn'
#endif

* calling argument declarations:
	INTEGER		lun
	CHARACTER*(*)	string

* internal variable declarations:
	LOGICAL 	multiline
	INTEGER		TM_LENSTR1, cr_pos, c1, clast

* initialize
	c1 = 1
	
* multiple lines ?
 100	cr_pos = INDEX( string(c1:), pCR )
	multiline = cr_pos .NE. 0

* get last character of line
	IF ( multiline ) THEN
	   clast = c1 + cr_pos - 2
	ELSE
	   clast = (c1-1) + TM_LENSTR1( string(c1:) )		! mod 2/90
	ENDIF

* write one line
#ifdef NO_PASSED_CONCAT
	risc_buff = string(c1:clast)
	CALL SPLIT_LIST(0, lun, '          '//risc_buff, 0)
#else
	CALL SPLIT_LIST(0, lun, '          '//string(c1:clast), 0)
#endif

* more ?
	IF ( multiline ) THEN
	   c1 = clast + 2
	   GOTO 100
	ENDIF

* no. done
	RETURN
	END
