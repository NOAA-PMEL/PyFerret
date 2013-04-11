	SUBROUTINE init_random_seed(inseed)
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
* Set seeds for calls to RANDOM_NUMBER.
* Both RANDOM_NUMBER and RANDOM_SEED are part of the gfortran library.
*
* This routine based on init_random_seed from the gfortran documentation
* except that the input argument inseed determines seeding as follows:
*  inseed = 0 return. Continue to use the seeding that's currently in effect
*  inseed = -1 use the system clock to initialize the seed
*  inseed = N > 0 use N to initialize the seed, for repeatable results.

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*


	INTEGER :: inseed
	INTEGER :: i, n, clock
	INTEGER, DIMENSION(:), ALLOCATABLE :: seed

	INTEGER iseed

	iseed = inseed
	IF (inseed .EQ. 0) RETURN

	CALL RANDOM_SEED(size = n)
	ALLOCATE(seed(n))

	IF (inseed .EQ. -1) THEN
	   CALL SYSTEM_CLOCK(COUNT=clock)
	   iseed = clock
	ENDIF
	seed = iseed + 37 * (/ (i - 1, i = 1, n) /)
	CALL RANDOM_SEED(PUT = seed)

	DEALLOCATE(seed)
	END SUBROUTINE
