	INTEGER FUNCTION REGION_CLASS( idim, cx )

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
* determine the region-limits class of a component in an expression
* the region-limits class encodes how limits must be specified in order to
* match with memory-resident data.  (It is uvar_given without the 
* "need"/"given" information.)

* note: the classes that exist as of FERRET 2.00 are complete
*	class 3 ("xact") assumes that any time both upper and lower limits must 
*	match then this must be a compressing transformation.  There should
*	really be a class "xact" to insist on exact match, uncompressed and a
*	class "pt" that compresses as well.  This would require updating the
*	tables in NEST_REGION_CLASS and MERGE_GRID.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/18/89
* V230:  1/27/92 - changed uvlim_*_hi (unused) to uvlim_*_lohi to describe
*                  @WEQ - exaxt limits required but a range of data
*                - also added @RSUM (like @IIN)
	
#ifdef unix
	include 'ferret.parm'
	include	'xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	idim, cx

	IF ( cx_trans( idim, cx ) .GT. trans_compress_code ) THEN
	   REGION_CLASS = 3    ! uvlim_*_xact
	ELSEIF ( cx_trans( idim,cx ) .EQ. trans_integ_indef
     .      .OR. cx_trans( idim,cx ) .EQ. trans_run_sum ) THEN
	   REGION_CLASS = 1    ! uvlim_*_lo
	ELSEIF ( cx_trans( idim,cx ) .EQ. trans_wgted_eq ) THEN
           REGION_CLASS = 2    ! uvlim_*_lohi
	ELSEIF ( cx_given( idim, cx )
     .	   .AND. cx_lo_ss(cx,idim) .EQ. cx_hi_ss(cx,idim) ) THEN
* ... e.g. U[K=3] ... defines a fixed point
	   REGION_CLASS = 3
	ELSE
	   REGION_CLASS = 0  
	ENDIF

	RETURN
	END
