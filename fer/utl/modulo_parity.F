	INTEGER FUNCTION MODULO_PARITY(ss1, modlen, delta)

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
* Determine the "parity" that is associated with the starting subscript ss1
* given a modulo length of lenmod and a stride value of delta
* for example, if the modulo length is 10
*	i=1:10:3  - parity = 1  ... ditto i=4:10:3
*	i=2:10:3  - parity = 2  ... ditto i=5:10:3
*	i=3:10:3  - parity = 3  ... ditto i=6:10:3
* Although i=11 would seem to be parity=2, it is the starting index of the
* next modulo replication so it is parity=1. i=12 is parity=2, etc.

* V533: *sh* 6/01
* V56: *acm* 3/04 Fix bugs that occur when ss1 is negative.

* calling argument declarations
	INTEGER	ss1, modlen, delta

* internal variable declarations
	INTEGER TM_MOD, nmod

	nmod = (ss1-1)/modlen  ! number of modulo replications preceding ss1

* Start at the bottom of the modulo replication preceding ss1
        IF (ss1 .LT. 0 .AND. nmod*modlen .NE. ss1-1) nmod = nmod - 1

	MODULO_PARITY = TM_MOD(ss1-1-(nmod*modlen), delta) + 1

	RETURN
	END
