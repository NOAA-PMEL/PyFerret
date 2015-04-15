	BLOCK DATA pyfont_block_data

C	Initialization of pyferret font arguments.
C  pyfont		Flag: True for PyFerret true fonts, False to use Hershey fonts
C  pyfnt_fontname	name of font
C  pyfnt_italic		integer: 1=italic, 0=not
C  pyfnt_bold		integer: 1=bold, 0=not
C  pyfnt_under		integer: 1=underscore, 0=not
C  pyfnt_red
C  pyfnt_grn		Real: percent Red, Green, Blue (range 0-100)
C  pyfnt_blu
C  pyfnt_opq		Real: percent opacity (range 0-100)
C  pyfnt_rot		Real: rotation angle

	IMPLICIT NONE

	INCLUDE 'pyfonts.cmn'

	DATA pyfont/.TRUE./
	DATA pyfnt_fontname/'Times New Roman'/
	DATA pyfnt_dflt/'Times New Roman'/
	DATA pyfnt_italic/0/
	DATA pyfnt_bold/0/
	DATA pyfnt_under/0/
	DATA pyfnt_red/0./
	DATA pyfnt_grn/0./
	DATA pyfnt_blu/0./
	DATA pyfnt_opq/1./
	DATA pyfnt_rot/0./

	END
