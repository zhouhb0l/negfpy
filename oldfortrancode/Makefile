#compiler
 F90 = ifort -O3
# LIBS = -L$MKL_LIB -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
#LIBS = -llapack
LIBS = -mkl
# LIBS = -Wl,--start-group /share/intel/mkl/10.1.0.015/lib/em64t/libmkl_intel_lp64.a /share/intel/mkl/10.1.0.015/lib/em64t/libmkl_sequential.a /share/intel/mkl/10.1.0.015/lib/em64t/libmkl_core.a -Wl,--end-group

 FFLAGS =

SUBDIRS = 
VPATH := $(SUBDIRS)
modinterf := sizes.mod material_para.mod running_para.mod force_constants.mod interfaces.mod
modobj := parameters.o
source := sgf.f90 sfgsolve.f90 transform.f90 force.f90\
          Transmission.f90 NEGF.f90
          
objects := ${source:.f90=.o}

#.PHONY: all clean
all: a.out

a.out: $(objects)
	$(F90) $(FFLAGS) -o $@ $^ $(modobj) $(LIBS)

%.o: %.f90 $(modinterf)
	$(F90) -c $(FFLAGS) $< -I..

$(modinterf):  parameters.f90 
	$(F90) -c $<

clean: 
	-rm  *.mod *.o

