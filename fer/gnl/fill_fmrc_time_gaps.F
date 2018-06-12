	SUBROUTINE FILL_FMRC_TIME_GAPS (has_gaps, time2d, edge2d,
     .					nt, nf, lagF, status)

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
* NOAA/PMEL, Seattle, WA - TMAP 

* if there are gaps in the forecast array that was provided, there will be
* missing rows in the 2d time array.  This routine fills thos gaps.
*
* V698 *sh* 12/15 
* V710 *acm* 1/17 Fix ticket 2500: TM_DFPEQ to handle tiny precision differences

* calling arguments
*   has_gaps - (in) flag if gap-filling is needed
*   time2d - (in) 2d forecast time step array (with missing time series)
*   edge2d - (in) 2d forecast time lower bound array (also with missing points)
*   nT	   - (in) number of time points in each forecast
*   nF	   - (in) number of forecasts
*   lagF   - (in) number of model time steps of lag between forecasts (output)

* calling argument declarations:
	LOGICAL has_gaps
	INTEGER nT, nF, lagF, status
	REAL time2d(nT, nF), edge2d(nT, nF)

* internal variable declarations:
	LOGICAL TM_DFPEQ
	INTEGER ifcst, istep, ifcst2, istep2
	REAL	tref
	CHARACTER LEFINT*8

	include 'ferret.parm'
	include 'errmsg.parm'
	
* initialize
	status = ferr_ok

* determine the number of indices lagged between rows
* the first and last time series are guaranteed to be present
* find the column where it is first found in a subsequent row
	tref = time2d(nT,1)
	DO ifcst = 2, nF
* ... find the next forecast that exists
	   IF (time2d(1,ifcst) .EQ. unspecified_val8) CYCLE
* ... find the time step of this forecast where tref gets repeated
	   DO istep = nT, 1, -1
	      IF (TM_DFPEQ(time2d(istep, ifcst), tref)) THEN
	         lagF = (nT-istep)/(ifcst-1)
	         GOTO 100
	      ENDIF
	   ENDDO
	   GOTO 5100  ! tref not found in this forecast
	ENDDO
	GOTO 5100     ! should never get here

 100	IF (.NOT.has_gaps) RETURN    ! we computed lagF.  Thats enuf

* fill in the missing time step values from adjoining rows
	DO ifcst = 2, nF
* ... skip valid forecasts
	   IF (time2d(1,ifcst) .NE. unspecified_val8) CYCLE
* ... copy the upper times from the preceding forecast
	   DO istep = lagF+1, nT
	      time2d(istep-lagF, ifcst) = time2d(istep, ifcst-1)
	      edge2d(istep-lagF, ifcst) = edge2d(istep, ifcst-1)
	   ENDDO
* ... look ahead to find the next valid forecast
	   DO ifcst2 = ifcst+1, nF
	      IF (time2d(1,ifcst2) .NE. unspecified_val8) GOTO 110
	   ENDDO
	   GOTO 5100  ! can never get here cuz the final row always exists
* ... copy the lower times from this future forecast
 110	   istep2 = (ifcst2-ifcst)*lagF
	   IF (istep2 .GT. nT-lagF) GOTO 5100  ! void too long
	   DO istep = nT-lagF+1, nT
	      time2d(istep,ifcst) = time2d(istep-istep2, ifcst2)
	      edge2d(istep,ifcst) = edge2d(istep-istep2, ifcst2)
	   ENDDO
	ENDDO

* success
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_aggregate_error, status,
     .		'time steps corrupted or '//
     .		'too many missing forecasts in a row at forecast '//
     .		LEFINT(ifcst, ifcst2), *5000 )

	END

