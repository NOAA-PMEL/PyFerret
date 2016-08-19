((fortran-mode . ((eval . (progn
			    (fortran-line-length 132)
			    (set (make-local-variable 'flycheck-ifort-include-path)
				 (append flycheck-ifort-include-path
					 '("/usr/local/src/pyferret_src/fer/common"
					   "/usr/local/src/pyferret_src/ppl/tmap_inc"
					   "/usr/local/src/pyferret_src/ppl/include")))
			    (set (make-local-variable 'flycheck-ifort-args) '("-fpp" "-r8" "-132" "-Difort" "-Ddouble_p"))
			)))))
