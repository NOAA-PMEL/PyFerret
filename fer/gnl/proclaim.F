	SUBROUTINE PROCLAIM ( logical_unit, left_margin )

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
* introduce the program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* V400 - 3/95 - changed "Revision" to "Version"
* V402 - 8/95 - use XREVISION_TYPE to automate version info
* V510 6/00 *sh* - if revision_type is "()" ignore it

#ifdef unix
	include 'xrevision.cmn'
	include 'xrevision_type.cmn'
	include 'xtoday.cmn'
	include 'xmake_date.cmn'
#else
	INCLUDE 'FERRET_CMN:XREVISION.CMN'
	INCLUDE 'FERRET_CMN:XREVISION_TYPE.CMN'
	INCLUDE 'FERRET_CMN:XTODAY.CMN'
	INCLUDE 'FERRET_CMN:XMAKE_DATE.CMN'
#endif

* calling argument declarations:
	INTEGER		logical_unit
	CHARACTER*(*)	left_margin

* local variable definitions
	INTEGER TM_LENSTR1

* get revision type info (8/95 - kind of a hack)
	IF ( revision_type .EQ. '()' ) THEN
	   progname_mod = ' '
	ELSE
	   progname_mod = revision_type
	ENDIF
	len_progname_mod = TM_LENSTR1(progname_mod)

	WRITE  ( logical_unit, 3000)
     .				left_margin,
     .				left_margin, program_name(1:len_program_name),
     .					     progname_mod(1:len_progname_mod),	
     .				left_margin, revision_level, make_date,
     .				left_margin, today_date, today_time

 3000	FORMAT ( 1X,A,'NOAA/PMEL TMAP'/
     .		 1X,A,'Program ',A,1X,A/
     .		 1X,A,'Version',F5.2,' - ',A/
     .		 1X,2A,1X,A/ )

	RETURN
	END
