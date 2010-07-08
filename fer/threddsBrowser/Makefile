#
# Makefile for creating threddsBrowser.jar
#
# JAVA_HOME must be defined as the location of a Java 6 Development Kit
# If not defined, this Makefile will attempt to define it if a directory 
# matching /usr/local/jdk1.6* is found.
#
# Also the TOOLSUI_JAR file define below must exist
#

TOOLSUI_JAR = toolsUI/toolsUI-4.1.jar

JAVA_SOURCE_FILES = $(wildcard src/gov/noaa/pmel/ferret/threddsBrowser/*.java)

ifndef JAVA_HOME
GUESS_JDK16_HOMES = $(shell ls -1 -r -d /usr/local/jdk1.6*)
ifneq ($(words $(GUESS_JDK16_HOMES)), 0)
JAVA_HOME = $(firstword $(GUESS_JDK16_HOMES))
endif
endif

.PHONY : all
all : threddsBrowser.jar

.PHONY : clean
clean : 
	rm -fr threddsBrowser.jar bin/*

threddsBrowser.jar : $(JAVA_SOURCE_FILES) Makefile
ifdef JAVA_HOME
	rm -fr threddsBrowser.jar bin/*
	$(JAVA_HOME)/bin/javac -sourcepath src -classpath bin:$(TOOLSUI_JAR) -d bin $(JAVA_SOURCE_FILES)
	$(JAVA_HOME)/bin/jar cf threddsBrowser.jar -C bin gov/noaa/pmel/ferret/threddsBrowser
else
	@echo "JAVA_HOME must be defined as the location of a Java 6 Development Kit"
	@echo "for example, /usr/local/jdk1.6.0_11 or /home/smith/jdk1.6.0_20"
	@exit 1
endif

