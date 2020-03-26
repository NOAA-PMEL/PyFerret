	SUBROUTINE SETUP_ANNOTATION ( user, nouser, norm, xloc, yloc,
     .        halign, valign, angle, size, status)

* Create the PPL call to put an annotation on the page, convertint the
* requested location, angle, etc to locations on the page for the annotation
* type: /USER, /NOUSER, or /NORMalized 

        IMPLICIT NONE
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

	xpos = xloc
	ypos = yloc

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


 5400   CALL ERRMSG (ferr_invalid_command, status, 
     .          'Could not get axis lengths for label positioning', *5000)

	END
