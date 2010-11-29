/\/\*[ 	]*FORTRAN[ 	]*\*\//,/{/ {

    /FORTRAN/ {
	i\
	#FORTC_LINE
	=
    }

    s/\/\*.*\*\///g

    s/^[ 	]*//
    s/[ 	]*$//

    H

    /{/!{
	d
    }

    g

    s/,/ /g

    s/\n/ /g
    s/	/ /g
    s/   */ /g

    s/^ //
    s/ $//

    /\([a-zA-Z_][a-zA-Z_0-9]*\)[ ]*(/ {
	s//M4__PROTO(`\1',/
    }

    /\([^a-zA-Z_]\)int[ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`INTSTAR(\2)'/g
    }
    /\([^a-zA-Z_]\)integer[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`INTSTAR(\2)'/g
    }

    /\([^a-zA-Z_]\)float[ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`FLOATSTAR(\2)'/g
    }
    /\([^a-zA-Z_]\)real[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`FLOATSTAR(\2)'/g
    }

    /\([^a-zA-Z_]\)double[ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`DOUBLESTAR(\2)'/g
    }
    /\([^a-zA-Z_]\)doubleprecision[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`DOUBLESTAR(\2)'/g
    }

    /\([^a-zA-Z_]\)char[ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`STRING(\2)'/g
    }
    /\([^a-zA-Z_]\)character[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`STRING(\2)'/g
    }

    /\([^a-zA-Z_]\)void[ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`VOIDSTAR(\2)'/g
    }

    /\([^a-zA-Z_]\)\([a-zA-Z_][A-Za-z_0-9]*\)[ ]*([ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)[ ]*)[ ]*([ ]*)/ {
	s//\1`FUNCTION(\2,\3)'/g
    }
    /\([^a-zA-Z_]\)\([a-zA-Z_][A-Za-z_0-9]*\)[ ]*function[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`FUNCTION(\2,\3)'/g
    }

    /\.\.\./ {
	s//`VARARGS'/g
    }

    /\([^a-zA-Z_]\)\([a-zA-Z_][a-zA-Z_]*\)[ ]*\*[ ]*\*[ ]*\([A-Za-z_][A-Za-z_0-9]*\)/ {
	s//\1`POINTER(\2,\3)'/g
    }

    s/{/M4__BODY/

    s/, /,/g
    s/( /(/g
    s/ )/)/g
    s/' `/'`/g

    x
    s/.*//g
    x

    p

    i\
    #FORTC_LINE
    =

    d
}
