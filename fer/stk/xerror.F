	SUBROUTINE XERROR(MESSG,NMESSG,NERR,LEVEL)

* dummy error routine for CMLIB package
* used to avoid changes to UNI.FOR

	CHARACTER*(*) messg
	INTEGER	NMESSG,NERR,LEVEL

	STOP 'UNI error'
	END
