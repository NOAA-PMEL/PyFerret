	CHARACTER*6 FUNCTION TM_EXP_CODE ( ds_num )
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
*
*  Returns a unique exp_code for use in looking at experiment documentation
*  files. This code is based on ds_type and ds_expnum. This is 12222, where
*  1 and 2 indicate one character of ds_type and ds_expnum respectively.
*  E.g. "G0023". Special cases exist for GTBA011, GTMA011, and GTPA011.
*  These will have B,M, or P appended as appropriate.
*
*  Arguments:
*	ds_num - dataset number
*
*  Written 04/88 by M. Verschell for PMEL/TMAP (for VAX/VMS)
*
* revision 0.00 - 04/28/88 - 
* revision 0.10 - 01/24/89 - Added patch so that TS data sets give G prefix
* revision 0.11 - 02/15/89 - removed ds_open
* Unix/RISC port 4/24/91 *sh*: increased ds_name size
* MOM change - 4/92- *kob* : Allowed for M**** to be an acceptable exp num.
* bug fix - 7/92- Check d_source instead of d_title for MOM data - oops*kob*
* efficiency addition - 7/92-added a check for a modnum of BC which - *kob* 
* 7/95 *sh* have "CDF" map into "M"
* 9/95 *sh* - remove special "BURST" kludge of Ben Giese's - some of the recent
*		runs contain the word "Burst" in their titles, too

*  Argument definitions
	INTEGER*4	ds_num

*  Include files
#ifdef unix
	include	'tmap_errors.parm'
	include 'xio.parm'
#include "tmap_dset.parm"
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XIO.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

*  Local definitions
	CHARACTER*6	temp_code
	INTEGER*4	i, istat
	INTEGER*4	STR_UPCASE
	LOGICAL		TM_HAS_STRING

	temp_code(1:6) = ' '

*  Is the requested data set initialized?
	IF (ds_name(ds_num) .EQ. char_init40) GOTO 9000

*  If data set type is 'GT' then check for burst experiment
* add a check for model runs using MOM (new, vibrant cray code) 4/92 kob
	IF (TM_HAS_STRING(ds_type(ds_num),'GT')
     .  .OR.TM_HAS_STRING(ds_type(ds_num),'CDF')) THEN		! 7/95
!	  IF (TM_HAS_STRING(ds_title(ds_num),'BURST')) THEN
!	     temp_code(1:1) = 'B'
	  IF (TM_HAS_STRING(ds_source(ds_num),'MOM')) THEN
             temp_code(1:1) = 'M'
          ELSE
	     temp_code(1:1) = 'G'
	  ENDIF
	ELSE
	  DO i = 1,4
	    IF (ds_type(ds_num)(i:i) .NE. ' ') THEN
	      istat = STR_UPCASE(TEMP_CODE(1:1),ds_type(ds_num)(i:i))
	      GOTO 100
	    ENDIF
	  ENDDO
	ENDIF

  100	istat = STR_UPCASE(temp_code(2:5),ds_expnum(ds_num))

*  Special cases
	IF (TM_HAS_STRING(ds_expnum(ds_num),'11')) THEN
	  IF (TM_HAS_STRING(ds_modnum(ds_num),'BA')) temp_code(6:6)='B'
	  IF (TM_HAS_STRING(ds_modnum(ds_num),'MA')) temp_code(6:6)='M'
	  IF (TM_HAS_STRING(ds_modnum(ds_num),'PA')) temp_code(6:6)='P'
	  IF (TM_HAS_STRING(ds_modnum(ds_num),'BC')) temp_code(6:6)='P'
	ENDIF

*  Time Series fix
	IF (TM_HAS_STRING(ds_type(ds_num),'TS')) temp_code(1:1) = 'G'

	TM_Exp_code = temp_code
	GOTO 9999

 9000	WRITE	( tooplu, * ) 'Unknown data set ',ds_num
	TM_Exp_code = ' '
	GOTO 9999

 9999	RETURN
	END
