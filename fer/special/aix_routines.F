* AIX replacement routine for CALL DATE(str) which
* is missing in IBM's xlf libraries

*	Steve Hankin 10/27/94
*
* *kob* 5/26/95	added ifdef HP_ADDS for gerror and flush which hp had prob's
*	        with

	SUBROUTINE AIX_DATE(str)

* calling argument declarations
	CHARACTER*(*) str

* local variable declarations
	CHARACTER aix_d_t*26

* call the AIX function
	CALL FDATE_(aix_d_t)

* dd-mmm-yy
	str = aix_d_t(9:10)//'-'//aix_d_t(5:7)//'-'//aix_d_t(23:24)

	RETURN
	END

***********************************************************************

* AIX replacement routine for CALL TIME(str) which
* is missing in IBM's xlf libraries

*	Steve Hankin 10/27/94

	SUBROUTINE AIX_TIME(str)

* calling argument declarations
	CHARACTER*(*) str

* local variable declarations
	CHARACTER aix_d_t*26

* call the AIX function
	CALL FDATE_(aix_d_t)

	str = aix_d_t(12:19)

	RETURN
	END

***********************************************************************

* AIX replacement routine for CALL FLUSH(lunit) which
* is missing in IBM's xlf libraries

*	Steve Hankin 10/31/94
*
*	*kob* 6/15/95 - for hp port -- changed lunit to 0 for fflush call
*
	SUBROUTINE FLUSH(lunit)

* calling argument declarations
	INTEGER lunit

#ifdef HP_ADDS
	call fflush(0)
#else
* call the AIX function
	CALL FLUSH_(lunit)
#endif
	RETURN
	END


***********************************************************************

* AIX replacement routine for CALL GETLOG(name) which
* is missing in IBM's xlf libraries

*	Steve Hankin 10/31/94

	SUBROUTINE GETLOG(name)

* calling argument declarations
	CHARACTER*(*) name        ! hope this is OK ??? CHAR*8 limit ?

* call the AIX function
	CALL GETLOG_(name)

	RETURN
	END

***********************************************************************

* AIX replacement routine for CALL FDATE(string) which
* is missing in IBM's xlf libraries

*	Steve Hankin 10/31/94

	SUBROUTINE FDATE(string)

* calling argument declarations
	CHARACTER*(*) string

* call the AIX function
	CALL FDATE_(string)

	RETURN
	END

#ifndef HP_ADDS
***********************************************************************

* AIX replacement routine for CALL GERROR(string) which
* is missing in IBM's xlf libraries

*	Steve Hankin 10/31/94

* Note that this routine does NOT attempt to reproduce the correct
*	behavior.  This is a matter of laziness because I could not think
*	of an easy way to create a system error that would be detected
*	at the TMAP library level from whence this routine is called

	SUBROUTINE GERROR(string)

* calling argument declarations
	CHARACTER*(*) string

* heres the cheap fix:
	string = 'AIX sys err msg inaccessible to FERRET'
	RETURN

***********************************************************************

	end
#endif
