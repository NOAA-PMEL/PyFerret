	logical function tm_number (string)

* Test to see if a string is a valid number
* J Davison 11.8.94

	integer		length, tm_lenstr1
	integer		result
	
#ifdef sun
        byte      cstring(255)
#else
        integer*1 cstring(255)
#endif

	character*(*)	string
	
**********************************************************************

* 	CONVERT TO VALID C STRING
	length = tm_lenstr1 (string)
	call tm_ftoc_strng (string, cstring, length+1)

*	CALL C ROUTINE TO READ STRING -- RETURNS 0 IF NOT A NUMBER
	call tm_number_sub (cstring, result)
	
	if (result .eq. 0) then
	   tm_number = .false.
	else
	   tm_number = .true.
	endif

	return
	end
