	SUBROUTINE SET_ANNOTATION (user, nouser, norm, xloc, yloc,
     .        halign, angle, size, status)

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
* Parse qualifiers for ANNOTATE and DEFINE ANNOTATION command, converting
* locations to the correct ones for /USER, /NOUSER, or /NORMalized specification.

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'

	LOGICAL user, norm, nouser
	INTEGER	status
	REAL	xloc, yloc, halign, angle, size

* local variable declarations:

	INTEGER	TM_LENSTR1,
     .		loc, rlen, item, itype, nchar, ierr
	REAL	val, xpos, ypos, valign, deg_to_rad, cos_ang, sin_ang, 
     .		a_dx, a_dy, hloc, xaxlen, yaxlen, rev, xrev, yrev, 
     .		xaxmin, xaxmax, yaxmin, yaxmax, xscale, yscale

	CHARACTER buff*20, symbuff*30

* plotting must already be turned on

	IF ( .NOT.pplus_started ) GOTO 5100

* initialize (size initialized in the calling routine)

	xpos = 0.
	ypos = 0.
	valign = -1.
	halign = -1.
	angle = 0.
	status = ferr_ok

* Get settings

* /USER /NOUSER or /NORM. Default is /USER
	user   = qual_given( slash_annotate_user )  .GT. 0
	nouser = qual_given( slash_annotate_nouser) .GT. 0
	norm   = qual_given( slash_annotate_norm  ) .GT. 0

	itype = 0
	IF (user) itype = itype+1
	IF (nouser) itype = itype+1
	IF (norm) itype = itype+1
	IF (itype .EQ. 0) user = .TRUE.
	IF (itype .GT. 1) GOTO 5200

* ... ANNOTATE/HALIGN=float
	loc = qual_given( slash_annotate_halign )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
	   buff = 'HALIGN'
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) halign = val
	ENDIF

* valign and halign will modify the location settings if /NORM
* ... ANNOTATE/VALIGN=float

	loc = qual_given( slash_annotate_valign )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
	   buff = 'VALIGN'
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) valign = val
	ENDIF

* ... ANNOTATE/XPOS=float
	
	loc = qual_given( slash_annotate_xpos )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
	   buff = 'XPOS'
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) xpos = val
	ELSE
	   GOTO 5300
	ENDIF

* ... ANNOTATE/YPOS=float
	loc = qual_given( slash_annotate_ypos )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
	   buff = 'YPOS'
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) ypos = val
	ELSE
	   GOTO 5300
	ENDIF

* ... ANNOTATE/ANGLE=float

	angle = 0.
	loc = qual_given( slash_annotate_angle )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
	   buff = 'ANGLE'
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) angle = val
	ENDIF

* ... ANNOTATE/SIZE=float
	loc = qual_given( slash_annotate_size )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
	   buff = 'SIZE'
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) size = val
	ENDIF

* Compute arguments to send to the PPL %LABEL command

* .../USER
	IF (user) THEN

* Get axis scaling.  For line plots and PLOT/SET the AXIS_MIN
* and AXIS_MAX symbols may not be set.

	   symbuff = 'XAXIS_MIN'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1)  THEN
	      symbuff = 'PPL$YMIN1'
	      CALL GETSYM( symbuff, buff, nchar, ierr)
	      IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   ENDIF
	   READ(buff,*) xaxmin

	   symbuff = 'XAXIS_MAX'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1)  THEN
	      symbuff = 'PPL$XMAX1'
	      CALL GETSYM( symbuff, buff, nchar, ierr)
	      IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   ENDIF
	   READ(buff,*) xaxmax
	   
	   symbuff = 'PPL$XLEN'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   READ(buff,*) xaxlen

	   symbuff = 'YAXIS_MIN'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1) THEN
	      symbuff = 'PPL$YMIN1'
	      CALL GETSYM( symbuff, buff, nchar, ierr)
	      IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   ENDIF
	   READ(buff,*) yaxmin

	   symbuff = 'YAXIS_MAX'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1) THEN
	      symbuff = 'PPL$YMAX1'
	      CALL GETSYM( symbuff, buff, nchar, ierr)
	      IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   ENDIF
	   READ(buff,*) yaxmax
	   
	   symbuff = 'PPL$YLEN'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   READ(buff,*) yaxlen

* is the axis reversed?

c	   symbuff = 'XAXIS_REVERSED'
c	   CALL GETSYM( symbuff, buff, nchar, ierr)
c	   IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
c	   READ(buff,*) rev
c	   IF (rev.EQ.0) xrev =  1
c	   IF (rev.EQ.1) xrev = -1

c	   symbuff = 'YAXIS_REVERSED'
c	   CALL GETSYM( symbuff, buff, nchar, ierr)
c	   IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
c	   READ(buff,*) rev
c	   IF (rev.EQ.0) yrev =  1
c	   IF (rev.EQ.1) yrev = -1

	   xrev = 1
	   yrev = 1
	   xscale = xrev*xaxlen/(xaxmax-xaxmin)
	   yscale = yrev*yaxlen/(yaxmax-yaxmin)

	   IF (halign .LE. -1) THEN
	      hloc = -1.
	   ELSEIF (halign .GE. 1) THEN
	      hloc = 1.
	   ELSE
	      hloc = 0.
	   ENDIF

	   deg_to_rad = 0.017453292
	   cos_ang = cos(angle*deg_to_rad)
	   sin_ang = sin(angle*deg_to_rad)
	   a_dx = ((hloc-halign)/2 * cos_ang + (1+valign)/2 * sin_ang) * size / xscale
	   a_dy = ((hloc-halign)/2 * sin_ang - (1+valign)/2 * cos_ang) * size / yscale

	   xloc = xpos + a_dx
	   yloc = ypos + a_dy

	   buff = 'USER'
	ENDIF

* .../NOUSER
	IF (nouser) THEN
	   
	   IF (halign .LE. -1) THEN
	      hloc = -1.
	   ELSEIF (halign .GE. 1) THEN
	      hloc = 1.
	   ELSE
	      hloc = 0.
	   ENDIF

	   deg_to_rad = 0.017453292
	   cos_ang = cos(angle*deg_to_rad)
	   sin_ang = sin(angle*deg_to_rad)
	   a_dx = ((hloc-halign)/2 * cos_ang + (1+valign)/2 * sin_ang) * size
	   a_dy = ((hloc-halign)/2 * sin_ang - (1+valign)/2 * cos_ang) * size

	   xloc = xpos + a_dx
	   yloc = ypos + a_dy

	   buff = 'NOUSER'
	ENDIF

* .../NORM
	IF (norm) THEN

* Get axis lengths. 
	   symbuff = 'PPL$XLEN'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   READ(buff,*) xaxlen
	   
	   symbuff = 'PPL$YLEN'
	   CALL GETSYM( symbuff, buff, nchar, ierr)
	   IF (nchar.EQ.0 .OR. ierr.EQ.1) GOTO 5400
	   READ(buff,*) yaxlen
	   
	   IF (halign .LE. -1) THEN
	      hloc = -1.
	   ELSEIF (halign .GE. 1) THEN
	      hloc = 1.
	   ELSE
	      hloc = 0.
	   ENDIF

	   deg_to_rad = 0.017453292
	   cos_ang = cos(angle*deg_to_rad)
	   sin_ang = sin(angle*deg_to_rad)
	   a_dx = ((hloc-halign)/2 * cos_ang + (1+valign)/2 * sin_ang) * size
	   a_dy = ((hloc-halign)/2 * sin_ang - (1+valign)/2 * cos_ang) * size

	   xloc = xpos*xaxlen + a_dx
	   yloc = ypos*yaxlen + a_dy

	ENDIF

	IF (halign .LE. -1) THEN
	   halign = -1.
	ELSEIF (halign .GE. 1) THEN
	   halign = 1.
	ELSE
	   halign = 0.
	ENDIF

 5000	RETURN

 5100   CALL ERRMSG (ferr_invalid_command, status, 
     .          'ANNOTATE: Plotting has not been started', *5000)
     
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'Specify only one of /USER /NOUSER /NORM', *5000 )

 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'/XPOS and /YPOS are required for ANNOTATE', *5000 )

 5400   CALL ERRMSG (ferr_invalid_command, status, 
     .          'Could not get axis lengths for label positioning', *5000)

	END
