#
# Makefile for creating threddsBrowser.jar
#

#
# Include ../site_specific.mk for the possible definition of JAVA_HOME
#
include ../site_specific.mk

TOOLSUI_JAR = toolsUI/toolsUI-4.1.jar

JAVA_SOURCE_FILES = $(wildcard src/gov/noaa/pmel/ferret/threddsBrowser/*.java)

.PHONY : all
all : threddsBrowser.jar

.PHONY : clean
clean : 
	rm -fr threddsBrowser.jar bin

threddsBrowser.jar : $(JAVA_SOURCE_FILES) Makefile
	rm -fr threddsBrowser.jar bin
	mkdir bin
ifdef JAVA_HOME
	$(JAVA_HOME)/bin/javac -sourcepath src -classpath bin:$(TOOLSUI_JAR) -d bin $(JAVA_SOURCE_FILES)
	$(JAVA_HOME)/bin/jar cf threddsBrowser.jar -C bin gov/noaa/pmel/ferret/threddsBrowser
else
	echo "JAVA_HOME not defined; building of threddsBrowser.jar skipped"
endif

