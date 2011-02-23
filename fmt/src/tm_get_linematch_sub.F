	SUBROUTINE TM_GET_LINEMATCH_SUB (gvn, tst, dup_name, dup_guts)

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
* associated with a "match" on lname. A match means that the the characters
* of lname are all found plus, optionally, digits. If no match is found
* the value unspecified_int4 is returned.

* dup_name - returned TRUE if there is an identical name match
* dup_guts - returned TRUE if there is a pattern name match AND the guts
*	     are identica


* V510 *sh* 4/00

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

* argument variables
	LOGICAL		dup_name, dup_guts
	INTEGER		gvn, tst

* local variable declarations
	LOGICAL		TM_SAME_LINE_DEF
	INTEGER*4	TM_LENSTR1, STR_CASE_BLIND_COMPARE
	INTEGER*4	gvn_len, tst_len, istat, ic
	CHARACTER	lname*128, c*1

* initialize
	dup_name = .FALSE.
	dup_guts = .FALSE.
	lname    = line_name(gvn)
	gvn_len  = TM_LENSTR1(lname)

* screen out obvious rejection criteria
	IF ( tst .EQ. gvn ) GOTO 1000	! don't compare to self
	IF ( line_name(tst)  .EQ. char_init16 ) GOTO 1000
	tst_len = TM_LENSTR1( line_name(tst) )
	IF ( tst_len .LT. gvn_len ) GOTO 1000

* test for base name match
	istat = STR_CASE_BLIND_COMPARE
     .			(line_name(tst)(:gvn_len),lname(:gvn_len))
	IF (istat .NE. str_match) GOTO 1000
	IF (tst_len .EQ. gvn_len) dup_name = .TRUE.

* test that only extra digits were added
	DO 90 ic = gvn_len+1, tst_len
	  c = line_name(tst)(ic:ic)
	  IF (  c .LT. '0' .OR. c .GT. '9' ) GOTO 1000
 90	CONTINUE

* found a name pattern match.  Now check if the guts match, too
	dup_guts = TM_SAME_LINE_DEF( gvn, tst )

* not a duplicate definition
 1000	RETURN

	END
