#
# Makefile for creating threddsBrowser.jar
#
# JAVA_HOME must be defined as the location of a Java 6 Development Kit
# If not defined, this Makefile will attempt to define it if a directory 
# matching /usr/local/jdk1.6* is found.
#
# Also the TOOLSUI_JAR file define below must exist
#

#
# For the definition of JAVA_HOME
#
include ../../site_specific.mk

TOOLSUI_JAR = toolsUI/toolsUI-4.1.jar

JAVA_SOURCE_FILES = $(wildcard src/gov/noaa/pmel/ferret/threddsBrowser/*.java)

.PHONY : all
all : threddsBrowser.jar

.PHONY : clean
clean : 
	rm -fr threddsBrowser.jar bin

threddsBrowser.jar : $(JAVA_SOURCE_FILES) Makefile
ifdef JAVA_HOME
	rm -fr threddsBrowser.jar bin
	mkdir bin
	$(JAVA_HOME)/bin/javac -sourcepath src -classpath bin:$(TOOLSUI_JAR) -d bin $(JAVA_SOURCE_FILES)
	$(JAVA_HOME)/bin/jar cf threddsBrowser.jar -C bin gov/noaa/pmel/ferret/threddsBrowser
else
	@echo "***"
	@echo "*** WARNING"
	@echo "*** JAVA_HOME must be defined as the location of a Java 1.6 Development Kit"
	@echo "*** for example, /usr/java/jdk1.6.0_22 or /usr/java/latest"
	@echo "*** optional threddsBrowser.jar not built (SET DATA /BROWSE won't work)"
	@echo "***"
endif

