	subroutine dummies_for_aix
!	entry fdate
!	entry flush
!	entry getlog
	entry gerror
!	entry tm_c_ver_name
!	entry tm_make_relative_ver
!	entry tm_c_rename

	end

#ifdef hp
	subroutine flush(lunit)
	call fflush(0)
	end
#endif
