	LOGICAL FUNCTION GKS_WINDOWS()

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
