	LOGICAL FUNCTION TM_CHECK_BOX_LOC (line, line_edges, npts)
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
* Checks to make sure a data point is enclosed by its box on an axis
*
* revision 0.00 - 05/26/88 - by M. Verschell for PMEL/TMAP
* V63  *acm* 10/09 Changes for gfortran build
* V72 6/17 *acm* for trac enhancement #767 -- dynamic coordinate storage 
*                line and line_edges are the coordinates and edges for this line
*
* Arguments
	REAL	line(*), line_edges(*)
	INTEGER	npts
*
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

	INTEGER	i, nothing
	LOGICAL	TM_DFPEQ
	REAL*8 a,b1,b2

	DO 100 i = 1,npts-1
	  a = line(i)
	  b1 = line_edges(i)
	  b2 = line_edges(i+1)
	  IF (a .LT. b1 .OR. a .GT. b2) THEN
            IF (TM_DFPEQ(a,b1) .OR. TM_DFPEQ(a,b2) ) THEN
               nothing = 0
            ELSE
	       TM_Check_box_loc = .FALSE.
	    ENDIF
	    RETURN
	  ENDIF
  100	CONTINUE

	TM_CHECK_BOX_LOC = .TRUE.
	RETURN
	END
