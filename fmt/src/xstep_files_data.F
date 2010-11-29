	BLOCK DATA XSTEP_FILES_DATA

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
* various pre-defined variables for COMMON/Xstep_files
* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 08/19/86
* revision 0.10 - 07/22/88 - Added TS variables
* revision 0.11 - 11/21/88 - variable initialization changed to char_initXX
* revision 0.12 - 04/24/91 - *sh* unix/RISC port: increased sf_name size
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* v581 *acm* 4/2005  Add sf_ndxs and sf_ndxe to use indices rather than timesteps
*                     for indexing mc datasets in mc_read.

#include "tmap_dset.parm"		
	include 'tmap_dims.parm'
	include 'xstep_files.cmn_text'


	DATA	sf_name			/maxstepfiles*char_init512/
	DATA	sf_setnum		/maxstepfiles*set_not_open/
	DATA	sf_aux_set_num		/maxstepfiles*set_not_open/
	DATA	sf_nrecperstep		/maxstepfiles*int4_init/
	DATA	sf_skip			/maxstepfiles*int4_init/
	DATA	sf_1step		/maxstepfiles*real8_init/
	DATA	sf_lastep		/maxstepfiles*real8_init/
	DATA	sf_delta		/maxstepfiles*real8_init/
	DATA	sf_reclen		/maxstepfiles*int4_init/
	DATA	sf_valid		/maxstepfiles*.FALSE./
	DATA	sf_lunit		/maxstepfiles*file_not_open/
	DATA	sf_index		/maxstepfiles*char_init23/
	DATA	sf_regvars		/maxstepfiles*.TRUE./
	DATA	sf_regtimes		/maxstepfiles*.TRUE./
	DATA	sf_ndxs 		/maxstepfiles*int4_init/
	DATA	sf_ndxe 		/maxstepfiles*int4_init/
	DATA	vf_firstvar		/maxvarfiles*int4_init/
	DATA	vf_lastvar		/maxvarfiles*int4_init/

	END
