	INTEGER FUNCTION TM_GET_LINENUM (user_line)
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
* Searches the list of pre-defined line_name's and returns the number
* associated with the line_name given by the user. If no match is found
* the value unspecified_int4 is returned.
*
* written 08/88 By M. Verschell for PMEL/TMAP
*
* revision 0.00 - 08/12/88
*
* *kob* 10/96 	- Linux port.  Linux didn't like tabs between "include"
*	          and what was being included.  removed them
* Include files
#ifdef unix
#include "tmap_dset.parm"	
	include 'tmap_dims.parm'	! 
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'	! pre-defined grid common
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'	! Data set constants
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'	! 
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'	! 
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'	! pre-defined grid common
#endif

	CHARACTER*(*)	user_line
	INTEGER*4	nline, istat
	INTEGER*4	STR_CASE_BLIND_COMPARE

	DO 100 nline=1,max_lines
	  istat = STR_CASE_BLIND_COMPARE(line_name(nline),user_line)
	  IF (istat .EQ. str_match) THEN
	    TM_GET_LINENUM = nline
	    GOTO 900
	  ENDIF
100	CONTINUE

	TM_GET_LINENUM = unspecified_int4

900	RETURN
	END
