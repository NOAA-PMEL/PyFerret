	SUBROUTINE GET_DELTA_CONTEXT ( slash_dx0, delta_given, status )

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
* interpret delta coordinate limits of the FORM DX
* ... slash_dx0 is the qualifier number of dx less 1
* qualifiers are assumed to be ordered dx,dy,dz,dt,di,dj,dk,dl

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/27/89
*	  5/2/90 - half extracted to APPLY_DELTA_CONTEXT
* V420 (IBM/AIX port of V411): 10/5/95 - buffer for "READ(var,*"
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* specifiers may give coordinate in subscript units (/DI,/DJ,/DK,/DL) or
* may give coordinate in "world" units (/DX,/DY,/DZ,/DT)
* the specifiers may be of the following forms:
*	/q=val			specifies a single delta
*	/q=val_1:val_2		specifies a separate delta for lowe and upper

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'errmsg.parm'	
	include 'ferret.parm'	
	include 'xcontext.cmn'
	include 'xdelta_context.cmn'
	include 'xprog_state.cmn'

* calling arguments:
	LOGICAL		delta_given
	INTEGER		slash_dx0, status

* internal variables:
	INTEGER		TM_LENSTR1, colon_pos, wworss, idim, iqual, loc, rlen
	CHARACTER	string*40, raw_string*44

* initialize
	delta_given = .FALSE.
	DO idim = 1, nferdims
 	   dcx_given( idim ) = .FALSE.
	ENDDO

* decode the delta qualifiers
	DO 200 wworss = 0, (nferdims+1), nferdims
	DO 100 idim = 1, nferdims

	   iqual = slash_dx0 + wworss + idim
	   loc = qual_given( iqual )
	   IF ( loc .EQ. 0 ) GOTO 100

	   delta_given = .TRUE.
	   IF ( dcx_given(idim) ) GOTO 5100
	   dcx_given(idim) = .TRUE.

* get the qualifier string
	   raw_string = cmnd_buff(qual_start(loc):qual_end(loc))
	   rlen = TM_LENSTR1(raw_string)
	   CALL EQUAL_STRING( raw_string, string, status )

* decode it
	   colon_pos = INDEX( string, ':' )
	   IF ( colon_pos .EQ. 0 ) THEN
	      READ( string, *, ERR=5200 ) dcx_lo(idim)
	      dcx_hi(idim) = dcx_lo(idim)		   
	   ELSE
*             Appending space just to get rid of a nonsense compiler warning
	      raw_string = string(:colon_pos-1)//' '
	      READ( raw_string, *, ERR=5200 ) dcx_lo(idim)	! AIX 10/95
	      raw_string = string(colon_pos+1:)
	      READ( raw_string, *, ERR=5200 ) dcx_hi(idim)	! AIX 10/95
	   ENDIF

* is it ww or ss ?
	   dcx_by_ss(idim) = wworss .EQ. nferdims

 100	CONTINUE
 200	CONTINUE

* success
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_relative_coord, status, 
     .		     'subscript and world position given on same axis: '//
     .		     raw_string(:rlen), *5000)
 5200	CALL ERRMSG( ferr_erreq, status, raw_string(:rlen), *5000)

	END
