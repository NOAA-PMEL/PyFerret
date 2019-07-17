 	CHARACTER*(*) FUNCTION AX_TITLE_UNITS( idim, grid, len )

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
* produce a title string with units suitable for an axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 8/25/88
* V320: 12/19/94 - recognize millibars and decibars as special axes
*		(kludge: use "m" for meters instead of exact unit string.
*			This because of "METERS" in the old grid files around.)
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V695+ *sh* 3/15 - generate axis titles for TIME LAG axes in FMRC variables
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	include 'ferret.parm'
	include 'xtext_info.cmn'

* calling argument declarations:
	INTEGER	idim, grid, len
	LOGICAL	GEOG_LABEL

* internal variable declarations:
        LOGICAL ITS_FMRC, ITSA_TRUEMONTH_AXIS
	INTEGER	TM_LENSTR1, i, line, ulen
	CHARACTER*46 unit_name

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

	line = grid_line( idim, grid )

	IF ( GEOG_LABEL( idim, grid ) ) THEN
	   AX_TITLE_UNITS = axis_title( idim )
	   len = axis_title_len( idim )
* special formatting for vertical axes - 12/19/94
	   IF ( idim .EQ. z_dim ) THEN
	      IF ( line_unit_code(line) .EQ. pun_millibars 
     .        .OR. line_direction( line ) .EQ. 'DU'     ) THEN
	         AX_TITLE_UNITS = "HEIGHT"
	         len = 6
	      ENDIF
	      IF ( line_unit_code(line) .EQ. pun_meters ) THEN
	         unit_name = 'm'		! kludge for "METERS"
	      ELSE
	         unit_name = line_units(line)
	      ENDIF
	      ulen = TM_LENSTR1( unit_name )
	      AX_TITLE_UNITS = AX_TITLE_UNITS(1:len)//' ('//unit_name(:ulen)//')'
	      len = len + 3 + ulen

	   ELSE	      

              unit_name = line_units(line)
	      IF ( idim .EQ. t_dim) unit_name = 
     .          unit_name(:TM_LENSTR1(unit_name))//' since '// line_t0(line)

* If its a true-month axis then make the output unit name be day
              IF ( ITSA_TRUEMONTH_AXIS(line) ) unit_name = 'day'
              ulen = TM_LENSTR1( unit_name )
              AX_TITLE_UNITS = AX_TITLE_UNITS(:len)//' ('//unit_name(:ulen)//')'
              len = len + 3 + ulen
	   ENDIF

	ELSE
	   AX_TITLE_UNITS = ww_dim_name( idim )
	   len = 1
	   IF ( line .NE. mnormal .AND. line .NE. munknown ) THEN
	      IF ( line_units(line) .NE. ' ' ) THEN
	         ulen = TM_LENSTR1( line_units(line) )
	         AX_TITLE_UNITS = AX_TITLE_UNITS(1:1)//' ('//line_units(line)(:ulen)//')'
	         len = 4 + ulen
	      ENDIF
	   ENDIF
	ENDIF

	RETURN
	END
