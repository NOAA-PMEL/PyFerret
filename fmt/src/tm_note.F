	SUBROUTINE TM_NOTE ( string, lunit )

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
* display a warning message to the operator

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 2/92 for TMAP library (from FERRET rouitine "WARN")
* *sh* 2/28/95 - allow message to be sent to alternative location
* *js* 12.12.97- changed to use tm_split_list
* 1/03 - *kob* g77 port - Need to use risc_buff for arbitrary concatenations
* 12/13 *sh* - enhancement to support multi-line outputs in one call by
*	embedding pCR characters into string as line separators
*       Note that the Ferret stack of output calls could use some cleanup since
*	Ferret lib routines call TMAP_FORMAT lib TM_SPLIT_MESSAGE which in turn
*	calls back into Ferret lib to SPLIT_LIST
*	WRITE_MULTI is a Ferret lib routine that has the functionality we need.
*	But it calls TM_SPLIT_MESSAGE. By calling it here we add to the mess 

* calling argument declarations:
        INTEGER lunit
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		TM_LENSTR1, slen

	include 'xalt_messages.cmn_text'
	include 'xrisc_buff.cmn'

* string length
	slen = TM_LENSTR1( string )

* do it
	IF ( alt_messages ) THEN
	   CALL TM_ALTERNATE_NOTE( string, lunit )
	ELSE
c	   WRITE ( lunit, 3000 ) string(:slen)
c 3000	   FORMAT ( ' *** NOTE: ',A )
	   risc_buff = string
	   CALL WRITE_MULTI(lunit, ' *** NOTE: '//risc_buff)
	ENDIF

	RETURN
	END
