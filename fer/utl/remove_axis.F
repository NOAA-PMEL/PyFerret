
	SUBROUTINE REMOVE_AXIS (axis)

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
* Deallocates one axis (line) slot, and pack the coordinate memory array.

* ACM 6/11/2002
* V5.41 7/30/02 ACM Fix def of npoints. Also, move the pointer into 
*                   line_mem array when the coordinates in line_mem have
*                   been changed. 

* Parameters and commons

	include	'tmap_dims.parm'
#	include	"tmap_dset.parm"
	include 'xtm_grid.cmn_text'
        include 'ferret.parm'

* Argument declaration

        INTEGER axis

* Local declarations
	INTEGER  iaxis, imem, first_pt, npoints, last_mem

        first_pt = line_subsc1 ( axis )
        npoints = 2*line_dim(axis) + 1

        last_mem = next_line_mem_pos - 1

*  Recover space in line_mem array that defines point locations, box sizes.

        IF (first_pt .NE. unspecified_int4  .AND. 
     .      last_mem .NE. unspecified_int4 ) THEN

           DO 200 imem = first_pt, last_mem
              line_mem(imem) = line_mem(imem+npoints)
  200      CONTINUE

        ENDIF

*  For all axes above this axis, whose line_mem entries were moved, 
*  also move the pointer into line_mem

        DO 300 iaxis = axis+1, line_ceiling 
           IF ( (.NOT. line_regular(iaxis) ) .AND. 
     .              (line_subsc1(iaxis) .NE. unspecified_int4) ) THEN

              IF (line_subsc1(iaxis) .GT. first_pt) 
     .           line_subsc1(iaxis) = line_subsc1(iaxis) - npoints

           ENDIF
  300   CONTINUE

        line_subsc1(axis) = unspecified_int4

        next_line_mem_pos = next_line_mem_pos - npoints

	RETURN
	END
