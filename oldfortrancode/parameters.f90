module sizes
integer :: Dimen 
integer :: ML,MR !number of degrees of freedom in a unit cell of Left and Right lead
integer :: ML0,MR0 !number of degrees of freedom in a unit cell of 0th layer of Left and right lead
integer :: N_C !number of layers in the center
integer, allocatable :: MC(:)
integer:: N_cut(2)
contains

subroutine input_size()
implicit none
character(3)::tmpc
integer :: i
open(201,file='para.in')
tmpc='str'
do while(tmpc .ne. '@Si')
read(201,*)tmpc
enddo
read(201,*)tmpc
read(201,*)dimen
read(201,*)tmpc
read(201,*)ML,MR,ML0,MR0,N_C
allocate(MC(N_C))
read(201,*)tmpc
read(201,*)(MC(i),i=1,N_C)
read(201,*)tmpc
read(201,*)N_cut(1), N_cut(2)
close(201)
end subroutine
end module


module material_para
double precision :: ax,ay !lattice constant
double precision :: kNN !nearest neighbour force constant
double precision :: a1(2),a2(2) !lattice vector 
double precision :: b1(2),b2(2) !Reciprocal lattice vector
double precision :: ma !the reference mass
double precision :: thickness
contains
subroutine get_material_para()
use sizes,only:dimen
character(4)::tmpc
double precision, parameter :: pi=3.14159265358d0;
open(201,file='para.in')
do while(tmpc.ne.'@Mat')
read(201,*)tmpc
enddo
read(201,*)tmpc
read(201,*)ax,ay
read(201,*)tmpc
read(201,*)ma
read(201,*)tmpc
read(201,*)thickness
close(201)

if(dimen.eq.3) then
a1=(/ax,0d0/);a2=(/0d0,ay/)
b1=2*pi/(a1(2)*a2(1)-a1(1)*a2(2))*(/-a2(2),a1(2)/);
b2=2*pi/(a1(1)*a2(2)-a2(1)*a1(2))*(/-a1(2),a1(1)/);
endif
if(dimen.eq.2) then
a1=(/ax,0d0/);a2=0d0;
b1=(/2d0*pi/ax,0d0/);b2=0d0;
endif

end subroutine

end module


module running_para
!define running parameters
double precision:: kmin(2),kmax(2),dk(2)
double precision :: wmin,wmax,dw
double precision :: eta
logical :: save_tkw = .false.
contains
subroutine read_running_parameters()
use material_para
use sizes, only:dimen
implicit none
double precision, parameter:: pi=3.14159265358d0
character(4):: tmpc
integer :: ios
integer :: save_tkw_flag
open(201,file='para.in')
do while(tmpc.ne.'@Run')
read(201,*)tmpc
enddo
read(201,*)tmpc
read(201,*)kmin(1),kmax(1),dk(1)
read(201,*)tmpc
read(201,*)kmin(2),kmax(2),dk(2)
read(201,*)tmpc
read(201,*)wmin,wmax,dw
read(201,*)tmpc
read(201,*)eta
save_tkw = .false.
save_tkw_flag = 0
read(201,*,iostat=ios) save_tkw_flag
if (ios.eq.0) then
  save_tkw = (save_tkw_flag.ne.0)
endif
close(201)
if(dimen.eq.2) then
kmin(2)=0d0;kmax(2)=0d0;dk(2)=1d0 !go inside the loop for once
endif

end subroutine

end module

module interfaces
interface
subroutine Transmission(ww,eta,N_kC,N_band,mass,selL,selR,gaL,gaR,Trans)
double precision :: ww,eta
integer :: N_kC !number of degrees of freedom in the center
integer :: N_band !number of bands of the wI-K-\Sigma
double precision :: mass(:)
double complex :: selL(:,:),selR(:,:),gaL(:,:),gaR(:,:)
double complex :: Trans
end subroutine

subroutine cal_sgf(w,eta,NX,NX0,k11,k01,k00,k12,gr)
implicit none
double precision :: w,eta
integer :: NX,NX0
double complex :: k11(NX,NX),k01(NX0,NX),k00(NX0,NX0),k12(NX,NX)
double complex :: gr(NX0,NX0)

end subroutine
subroutine self_energy(layer,sgL,sgR,VLC,VCR,selL,selR,gaL,gaR)
implicit none
integer :: layer(:)
double complex :: sgL(:,:),sgR(:,:),VLC(:,:),VCR(:,:)
double complex :: selL(:,:),selR(:,:),gaL(:,:),gaR(:,:)
end subroutine

subroutine left_lead(w,eta,NX,NX0,k11,k01,k00,k12,gr,layer)
implicit none
double precision :: w,eta
integer :: NX,NX0
double complex :: k11(NX,NX),k01(NX0,NX),k00(NX0,NX0),k12(NX,NX)
double complex :: gr(NX0,NX0)
integer :: layer(:)
end subroutine

subroutine sfg_solve(GF,omega,h00,h11,h01)
implicit none
!calculate the surface green's function of a quasi-1d semi-infinite periodic
!system using the generalized eigen value method
!
!The arguments are the same as those of Wang Jian's implementation of
!recursive method:
!
!h00 is ((\omega+i\eta)^2-H00) for phonon problem, 
!while it is ((\varepsilon+i\eta)-H00) for electron problem.
!the same with H11
double complex, dimension(:,:), intent(in) :: H00,H01,H11
double complex, dimension(:,:), intent(out) :: GF 
double complex, intent(in) :: omega
!
end subroutine

end interface

end module






