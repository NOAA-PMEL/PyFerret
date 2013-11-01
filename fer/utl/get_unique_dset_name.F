	CHARACTER*(*) FUNCTION GET_UNIQUE_DSET_NAME ( dset )

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
* Return the short datset name, unless there is another dataset open with
* the identical name. In that case return the full name with path or url.

*  ACM 10/2013

        include 'tmap_errors.parm'      ! error codes
#include "tmap_dset.parm"
        include 'tmap_dims.parm'        ! data set dimensions
        include 'xdset_info.cmn_text'   ! data set common
        external xdset_info_data

* argument
	INTEGER dset

* local variable definitions
	INTEGER STR_SAME, slen, iset, ic
	CHARACTER*2048  dset_name


* get the short name of this dataset

	 dset_name = ds_name(dset)

* If no name, return.

	 GET_UNIQUE_DSET_NAME = dset_name
	 IF ( ds_name(dset) .EQ. char_init40 ) GOTO 120

         DO 110 iset = 1,maxdsets
           IF (ds_name(iset) .EQ. char_init1024) GOTO 110
	   IF (iset .EQ. dset) GOTO 110
	   ic = STR_SAME (dset_name, ds_name(iset))
           IF (ic .EQ. 0) THEN
	       GET_UNIQUE_DSET_NAME = ds_des_name(dset)
               GOTO 120
	   ENDIF
           
  110   CONTINUE

        GET_UNIQUE_DSET_NAME = dset_name

  120   CONTINUE
	RETURN
	END

