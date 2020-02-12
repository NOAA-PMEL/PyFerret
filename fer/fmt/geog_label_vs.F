	SUBROUTINE GEOG_LABEL_VS (buff, iunits, dir_dim, formatted_dim)

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
* This routine checks the units and the setting for formatted lon/lat (time?) axes
* If the formatting has been turned off, resets iunits and flag mod_vs_x.

* 3/2017 Ansley Manke
* for ticket 2517, correct labeling of lon/lat axes for PLOT/VS plots, and turn off
* lon/lat axis formatting according to MODE long_label and MODE latit_label
* for issue 1967, if the units are simply degrees or deg with no indication of 
* longitude or latitude, do not make formatted axis.  units="degrees" may be
* a compass direction.

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xprog_state.cmn'

	INTEGER iunits, dir_dim, formatted_dim
	CHARACTER*(*) buff
	

	LOGICAL TM_HAS_STRING

	formatted_dim = no_dim

* The units code has been set by a call to TM_FMT which already checks 
* whether the units string indicated degrees temperature.

* If the units are explicitly defined as degrees east or degrees north,
* set the formatted_dim flag.

	IF (iunits .EQ. 4) THEN

	   IF (TM_HAS_STRING(buff, '_e') .OR. TM_HAS_STRING(buff, ' e') 
     .         .OR. TM_HAS_STRING(buff, 'lon')) THEN
	     IF (ax_fmt(x_dim)) formatted_dim = x_dim
	     IF (.NOT. ax_fmt(x_dim)) iunits = 0
	   ELSEIF (TM_HAS_STRING(buff, '_n') .OR. TM_HAS_STRING(buff, ' n') 
     .         .OR. TM_HAS_STRING(buff, 'lat') ) THEN
	     IF (ax_fmt(y_dim)) formatted_dim = y_dim
	     IF (.NOT. ax_fmt(y_dim)) iunits = 0
	   ELSE

* If the lon or lat label mode has been turned off, unset the units 
	      IF (.NOT. ax_fmt(dir_dim)) iunits = 0

	      iunits = 0  ! require that for VS plots the units are explicitly east or north
	   ENDIF

	ENDIF

	RETURN
	END
