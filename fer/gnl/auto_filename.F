	CHARACTER*(*) FUNCTION AUTO_FILENAME( coding, cx, mr, flen )

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
* create a string describing the data set, region, transformations and variable
* that is suitable for a filename for listed output

* format is Wdddddddvvvv.iiiiijjjjjkkkkklllll
* where ddddd is the data set id
* 	vvvv identifies the variable or expression
* and	iiiii,jjjjj,kkkkk,lllll identify the axis regions and transformations

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/26/87
* revision 0.1 - 3/10/88 - eliminated HASH_NAME (moved to VAR_CODE)
* revision 0.2 - 6/27/88 - "coding" received in CALL instead of XPROG_STATE.CMN
* V200:  5/19/89 - 4D symmetrical
*	12/19/89 - eliminated the possibility of multiple data set variables
* Unix/RISC port - 9/91 - GET_DSET_NAME --> GET_SHORT_DSET_NAME (no path)
* SUN port - cleaned up the calculation of flen using risc_buff
* V230:  6/9/92 - distinct requirements for netCDF: "F" leadin character and
*                 extension ".cdf"
*               - also added protections against name overflowing buffer

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xprog_state.cmn'
        include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
        INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations
	CHARACTER*(*)	coding
	INTEGER		cx, mr, flen

* local parameter declarations:
	LOGICAL    no_decims
	PARAMETER( no_decims = .FALSE. )

* local variable declarations:
	LOGICAL	  include_dset, include_vcode, include_axis(4)
	CHARACTER LOHI_WORLD*24, LEGAL_CHARS*24, CX_DIM_STR*24,
     .		  LEFINT*4, SANITARY_VAR_CODE*4
	INTEGER TM_LENSTR,
     .		slen, leg_len, dset, idim, minus, mp
	
* determine special mods requested via "AUTO-q" where "q" is X,Y,Z,T,I,J,K,L,D,V
	minus = INDEX( coding, '-' )
	IF ( minus .EQ. 0 ) THEN
	   include_dset  = .TRUE.
	   include_vcode = .TRUE.
	   DO 10 idim = 1, 4
 10	   include_axis( idim ) = .TRUE.
	ELSE
	   mp = minus + 1
	   include_dset  = INDEX(coding(mp:), 'D' ) .EQ. 0
	   include_vcode = INDEX(coding(mp:), 'V' ) .EQ. 0
	   DO 20 idim = 1, 4
	   include_axis( idim ) = 
     .			INDEX(coding(mp:), ww_dim_name(idim) ) .EQ. 0
     .		  .AND. INDEX(coding(mp:), ss_dim_name(idim) ) .EQ. 0
 20	   CONTINUE
	ENDIF

* don't include information on axes that are irrelevant
	DO 30 idim = 1, 4
	  IF ( cx_lo_ww(idim,cx) .EQ. unspecified_val8 )
     .						include_axis(idim) = .FALSE.
 30	CONTINUE

* special lead-in characters (so DIR listings are neat)
        IF ( list_fmt_type .EQ. plist_unformatted ) THEN
           risc_buff = 'UNF'
           flen = 3
        ELSEIF ( list_fmt_type .EQ. plist_epic ) THEN
           risc_buff = 'EP'
           flen = 2
        ELSEIF ( list_fmt_type .EQ. plist_gt ) THEN
           risc_buff = 'GT'
           flen = 2
        ELSEIF ( list_fmt_type .EQ. plist_CDF ) THEN
           risc_buff = 'F'
           flen = 1
        ELSE
           risc_buff = 'W'
	   flen = 1
        ENDIF

* data set name or "nnnsets"
	IF ( include_dset ) THEN
	   dset = cx_data_set( cx )
	   IF ( dset .NE. pdset_irrelevant ) THEN
	      CALL GET_SHORT_DSET_NAME( dset, risc_buff(flen+1:), slen )
	      flen = flen + slen
	   ENDIF
	ENDIF

* variable code or Ennn ( random hashed number to represent the expression )
	IF ( include_vcode ) THEN
           risc_buff( flen+1:flen+4 ) = SANITARY_VAR_CODE
     .                          ( mr_category(mr), mr_variable(mr) )
	   flen = TM_LENSTR( risc_buff(:flen+4) )
	ENDIF

* netCDF filenames replace the elaborate AUTOFILENAME extension with ".cdf"
        IF ( list_fmt_type .EQ. plist_CDF ) THEN
           risc_buff = risc_buff(:flen) // '.cdf'
           flen = flen + 4
        ELSE
* non-netCDF filenames:
* point to separate name from extension
	   flen = flen + 1
	   risc_buff( flen:flen ) = '.'

* all relevant axis limits
	   DO 100 idim = 1, 4
	      IF ( include_axis( idim ) 
     .	  .   AND. cx_lo_ss( cx, idim ) .NE. unspecified_int4 ) THEN
	         risc_buff( flen+1:) = LEGAL_CHARS(
     .		   CX_DIM_STR(idim,cx,':',no_decims,slen), leg_len )
	         flen = flen + leg_len
	      ENDIF
 100	   CONTINUE
        ENDIF

* pass result back - overwrite garbage from "LEGAL_CHARS" with blanks
        risc_buff(flen+1:) = ' '
	AUTO_FILENAME = risc_buff

	RETURN
	END
