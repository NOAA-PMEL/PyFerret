	SUBROUTINE TM_CHECK_LINE_EDGES ( line, line_edges, npts, 
     .      iaxis, reversed, vname, vlen, is_double, regular, ok)

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

* Checks for edge coordinate data in a netCDF file

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG

* revision 0.0 - 6/17
* V72 6/17 *acm* for trac enhancement #767 -- dynamic coordinate storage 

* argument declarations

	LOGICAL reversed, is_double, its_irregular, regular, ok
	REAL line(*), line_edges(*)
	INTEGER npts, vlen, iaxis, bad_bndsid
	CHARACTER*(*) vname

* Local declarations	
	LOGICAL TM_CHECK_BOX_LOC, TM_CHECK_BNDS_CENTERED
	INTEGER i, irev, npp1
	REAL tmp8

* if the coordinates were reversed then the box edges must be, too
	IF ( reversed .AND.  npts .GT. 1) THEN
           npp1 = npts+1
	   DO i = 1, npp1/2
	      tmp8 = line_edges(i)
	      line_edges(i) = line_edges(npp1+1-i)
	      line_edges(npp1+1-i) = tmp8
	   ENDDO
	ENDIF

*  Checks to make sure coordinates are enclosed by boxes on the axis

        IF ( .NOT.TM_CHECK_BOX_LOC(line, line_edges, npts) ) THEN
           ok = .FALSE.
        ELSE 

* check to see that each data point is contained inside its box.
* See if they're centered after all.

           regular = TM_CHECK_BNDS_CENTERED(line, line_edges, npts, iaxis, is_double)
	   ok = .TRUE.
        ENDIF

	RETURN
	END
