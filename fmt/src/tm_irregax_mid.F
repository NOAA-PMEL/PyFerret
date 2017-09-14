	SUBROUTINE TM_IRREGAX_MID( line, line_edges, npts )

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
* Fill in the midpoint values as the default boundaries between grid
* boxes

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/93 - code lifted from cd_get_1_axis
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
*                The argument is the coordinates for just this line

* argument definitions:
*	line	 - coordinates of the line in dynamic memory
*	line_edges - coordinates of the bounds in dynamic memory
*	npt	 - length of axis

* argument declarations
	REAL	line(*), line_edges(*)
	INTEGER	npts

* internal variable declarations:
        INTEGER i

* include files
	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include 'xio.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* nominal treatment for one-point axis.

        IF (npts .EQ. 1) THEN
           i = 1
	   line_edges(i) = line(i) - 1. 
           line_edges(i+1) = line(i) + 1. 
	   RETURN
	ENDIF

* compute midpoints lying between coordinates
        DO i = 1, npts-1
           line_edges(i+1) = (line(i)+line(i+1)) / 2.0D0 
	ENDDO

* lower bound of first box
        line_edges(1)   = line(1) - (line(2)-line(1))/2.D0

* upper bound of last box
        line_edges(npts+1) = line(npts) + (line(npts)-line(npts-1))/2.D0

        RETURN
	END
