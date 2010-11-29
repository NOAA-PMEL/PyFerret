	SUBROUTINE GET_SHORT_DSET_NAME( dset, name, len_text )

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
* get the name, only - not the full path) of the given data set
* if that is meaningful
* else get a string describing the nature of the name 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* Unix/RISC port 5/2/91 *sh*: based on GET_DSET_NAME
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V510: *sh* 3/00 - IRI Web server trap - if "dods" is the dset name then
*		give long version 

#ifdef unix
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER		dset, len_text
	CHARACTER*(*)	name

* internal variable declarations:
	INTEGER		TM_LENSTR1, max_len_out, i1, slash

* internal parameter declarations
	INTEGER arb_outmax
	PARAMETER (arb_outmax = 40)	! limit DODS output len

* initialize
	max_len_out  = LEN( name )

* consider the options
	IF     ( dset .EQ. pdset_irrelevant	)	THEN	! no name
	   name  = 'N/A'
	   len_text = 3

	ELSE
	   len_text = TM_LENSTR1( ds_name( dset ) )
           IF ( len_text .GT. max_len_out ) THEN
* ... too long to fit
              name = ds_name(dset)(len_text-max_len_out+1:)
              len_text = max_len_out
	   ELSEIF ( ds_name(dset) .EQ. 'dods'
     .	    .AND.   ds_des_name(dset)(1:7) .EQ. 'http://' ) THEN
	      len_text = TM_LENSTR1(ds_des_name(dset))
* ... find start of IRI name as in "http://ip.address/directory/name/dods"
	      slash = INDEX(ds_des_name(dset)(8:),'/')
	      slash = 8 + slash + INDEX(ds_des_name(dset)(8+slash:),'/')
	      IF (slash .GT. 0 ) THEN
	         i1 = slash + 1
	      ELSE
	         i1 = MAX(1,len_text-max_len_out+1)
	         i1 = MIN(i1,arb_outmax)
	      ENDIF
              name = ds_des_name(dset)(i1:len_text)
              len_text = len_text - i1 + 1
           ELSE
              name = ds_name( dset )
           ENDIF

	ENDIF

	RETURN
	END
