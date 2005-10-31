
	SUBROUTINE PACK_LINE_STORAGE (axis)

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
* For a line (axis) that has been deallocated, pack the coordinate memory array.

* ACM 6/11/2002
* V5.41 7/30/02 ACM Fix def of npoints. Also, move the pointer into 
*                   line_mem array when the coordinates in line_mem have
*                   been changed. 
* V5.90 10/28/05 ACM Fix bug 1354: We were moving way more coord data than needed
* v5.90 10/31/05 ACM Renamed from REMOVE_AXIS.F
*                   

* Parameters and commons

	include	'tmap_dims.parm'
#	include	"tmap_dset.parm"
	include 'xtm_grid.cmn_text'
        include 'ferret.parm'

* Argument declaration

        INTEGER axis

* Local declarations
	INTEGER  TM_GET_LINENUM, iaxis, imem, first_pt, last_pt, 
     .           npoints, n_to_move, num_protected_axes

	num_protected_axes = TM_GET_LINENUM( 'EZ' )  ! top protected axis

        IF (num_protected_axes .EQ. unspecified_int4 .OR.
     .      num_protected_axes .LE. 0) num_protected_axes = 1

* First index in this line, number of points in line

        first_pt = line_subsc1 ( axis )

*  Recover space in line_mem array that defines point locations, box sizes.
*  If this line is the last in the coordinate storage, move the pointer only.

        IF (first_pt .NE. unspecified_int4  .AND. 
     .      first_pt .NE. int4_init) THEN

           npoints = 2*line_dim(axis) + 1
           last_pt = first_pt + npoints

           n_to_move = next_line_mem_pos - last_pt

           IF (last_pt  .LT. next_line_mem_pos) THEN
              DO 200 imem = 1, n_to_move 
                 line_mem(first_pt+imem-1) = line_mem(last_pt+imem-1)
  200         CONTINUE
           ENDIF

*  For all axes whose line_mem entries were moved, also move the pointer 
*  into line_mem

           DO 300 iaxis = num_protected_axes+1, line_ceiling 
              IF ( (.NOT. line_regular(iaxis) ) .AND. 
     .              (line_subsc1(iaxis) .NE. unspecified_int4) ) THEN

                 IF (line_subsc1(iaxis) .GT. first_pt) 
     .            line_subsc1(iaxis) = line_subsc1(iaxis) - npoints

              ENDIF
  300      CONTINUE
  
           line_subsc1(axis) = unspecified_int4

           next_line_mem_pos = first_pt + n_to_move 

        ENDIF

	RETURN
	END
