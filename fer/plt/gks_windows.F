	LOGICAL FUNCTION GKS_WINDOWS()

* determine if the GKS is installed on the system and can be used from the
* current environment
* ... as of 5/01/91 the latter requirement isn't really checked ...

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 8/16/88
* V200:  9/12/89 - DECwindows
*	 1/25/90 - based on "GKS_WSTYPE" instead of "TT"
*	 5/01/91 - unixify *JD*
* V230: simply return .TRUE. on Unix systems

#ifdef unix
      GKS_WINDOWS = .TRUE.
!* internal variable declarations:
!	LOGICAL		there
! 
!* search for GKS3D library
!	INQUIRE (FILE='/lib/libGKS3D.a',EXIST=there)	
!
!* if there then GKS is installed
!	GKS_WINDOWS = there
#else
	INCLUDE '($SSDEF)'                              ! LIB_ return status

* internal variable declarations:
	INTEGER		LIB_SYS_TRNLOG, stat, dlen
	CHARACTER*24	transl

* translate the logical name used as a flag to see if GKS is on the system
	stat = LIB_SYS_TRNLOG( 'GKS_WSTYPE', dlen, transl )

* "normal" means the logical exists
	GKS_WINDOWS = stat .EQ. ss$_normal
#endif	

	RETURN
	END
