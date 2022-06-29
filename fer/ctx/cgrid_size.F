	INTEGER FUNCTION CGRID_SIZE ( cx )

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
* determine the number of values in a 2D grid that lies within the plane
* of the specified context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/24/86
* revision 0.1 - 7/20/87 - kludge introduction of 1D grids for XY averages
* V200:  5/18/89 - 4D symmetrical
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* v767 6/22 *acm* See issue 112: Unexpected error: this routine called
*		  when the grid is too large for integer*4 indexing

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'

	INTEGER	CX_DIM_LEN, cx
	INTEGER*8 CGRID_SIZE8, gsize  
	INTEGER tsize

	gsize = CGRID_SIZE8( cx )
	IF (gsize .GT. arbitrary_large_int4) THEN
	   CALL WARN ('CGRID_SIZE: Overflow in computing size of grid')
	   CALL WARN ('Internal Error: Exiting')
	   STOP
	ENDIF

	tsize = CX_DIM_LEN( 1, cx )
     .		   * CX_DIM_LEN( 2, cx )
     .		   * CX_DIM_LEN( 3, cx )
     .		   * CX_DIM_LEN( 4, cx )
     .		   * CX_DIM_LEN( 5, cx )
     .		   * CX_DIM_LEN( 6, cx )

	CGRID_SIZE = tsize

	RETURN
	END
