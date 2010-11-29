	INTEGER FUNCTION TM_AXIS_STRIDE (axis, i1)

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
* Return the "stride" value and starting index for an axis that is the
* child of another

* V533 *sh* 6/01

	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'xtm_grid.cmn_text'

* calling argument declarations
	INTEGER axis, i1

* internal variable declarations
	REAL	parent_delta

	IF ( line_class(axis) .EQ. pline_class_stride ) THEN

	   IF ( line_regular(axis) ) THEN
	      parent_delta = line_delta(line_parent(axis))
	      TM_AXIS_STRIDE = INT( 1.001 * line_delta(axis)/
     .			                    parent_delta )
	      i1 = INT( 1.001 *
     .			(line_start(axis) - line_start(line_parent(axis))) / 
     .			parent_delta ) + 1
	   ELSE
	      TM_AXIS_STRIDE = INT( line_delta(axis) )
	      i1 = INT( line_start(axis) )
	   ENDIF

	ELSE

	   TM_AXIS_STRIDE = 1
	   i1 = 1

	ENDIF

	RETURN
	END
