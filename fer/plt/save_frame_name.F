
	SUBROUTINE SAVE_FRAME_NAME(fname, clen)
	implicit none

	include 'tmap_dims.parm'
	include 'xprog_state.cmn'

	integer clen

#ifdef sun
	byte fname(clen) 
#else   
	integer*1 fname(clen)
#endif  
	call tm_ctof_strng(fname, frame_file, clen)

	return
	end

