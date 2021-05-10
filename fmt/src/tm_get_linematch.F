	INTEGER FUNCTION TM_GET_LINEMATCH (gvn_line, dup_name, dup_guts)

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

* dup_name - returned TRUE if there is an identical name match OR (there
*	     is a pattern name match AND the guts are identical)
* dup_guts - returned TRUE if there is a pattern name match AND the guts
*	     are identica


* V510 *sh* 4/00
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V6.95+ 11/15 *sh*  when dup_guts is found do not return immediately -- 
*                    instead continue through the axes to see if there may
*                    also be a dup_name. This for special needs of creating
*		     an aggregated T axis by modifying the axis of the 1st file
*                   

* Include files
#include "tmap_dset.parm"	
	include 'tmap_dims.parm'	
#include "gt_lib.parm"

* argument variables
	LOGICAL		dup_name, dup_guts
	INTEGER		gvn_line

* local variable declarations
	LOGICAL		xact_name
	INTEGER*4	i

* initialize
	dup_name = .FALSE.
	TM_GET_LINEMATCH = unspecified_int4

* Note: the code logic appears to return the *last* dup_guts found, but
*       logic in Ferret ensures that there can be at most one such match 

* check the static axes
	DO 100 i=1,max_lines
	  CALL TM_GET_LINEMATCH_SUB(gvn_line, i, xact_name, dup_guts)
	  IF (dup_guts) TM_GET_LINEMATCH = i
	  IF (xact_name) dup_name = .TRUE.
 100	CONTINUE
 
* check the dynamic grids
	i = 0
 200	CALL TM_NEXT_DYN_LINE( i, *299)
	  CALL TM_GET_LINEMATCH_SUB(gvn_line, i, xact_name, dup_guts)
	  IF (dup_guts) TM_GET_LINEMATCH = i
	  IF (xact_name) dup_name = .TRUE.
	GOTO 200

 299	RETURN
	END
