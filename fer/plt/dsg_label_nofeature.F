	SUBROUTINE DSG_LABEL_NOFEATURE ( dummy, labsize, status)

* 3/26/20 
* Use annotation-calls to put a label "No Profile Data in Selection" 
* across a plot where constraints resulted in no features selected.

        IMPLICIT NONE
	include 'errmsg.parm'

	CHARACTER*(*)	dummy
	REAL		labsize
	INTEGER		status

	REAL		xloc, yloc, halign, valign, angle
	INTEGER		nlab

	xloc = 0.5
	yloc = 0.5
	halign = 0.
	valign = 0.
	angle = 0.0
	nlab = 0  ! define next moveable label

c  first 3 argument are user, nouser, norm
	CALL SETUP_ANNOTATION (.FALSE., .FALSE., .TRUE.,  
     .         xloc, yloc, halign, valign, angle, labsize, status)

	IF (status .EQ. ferr_ok) CALL PUT_ANNOTATION (
     .         .FALSE., .FALSE., .TRUE., xloc, yloc, halign, 
     .         angle, labsize, nlab, dummy, status)

	RETURN
	END   
