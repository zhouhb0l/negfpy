program force
integer :: Na,Nx,Ny,Nz
double precision, allocatable :: f(:,:,:,:,:,:,:)


call read_QE(Na,Nx,Ny,Nz,f)
call analyze_f_1(Na,Nx,Ny,Nz,f)


contains

subroutine analyze_f_1(Na,Nx,Ny,Nz,f)
implicit none
integer :: Na,Nx,Ny,Nz
double precision :: f(:,:,:,:,:,:,:)
!local variables
integer :: MR
double precision,allocatable :: kR11(:,:,:,:),kR01(:,:,:,:)
integer :: N_cut(2)
integer :: i,j,k,l,m
integer :: icell,iatom,ixyz,jcell,jatom,jxyz
integer :: ncell !number of unit cells involved in one layer
integer :: xcell,ycell,zcell !The relative cell index between two atoms
!N_cut(1)=Ny;N_cut(2)=1;
N_cut(1)=Ny/2-1;N_cut(2)=0

ncell=nx/2
MR=3*Na*ncell
allocate(kR11(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MR,MR),kR01(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MR,MR))

do i=1,MR
do j=1,MR
icell=(i-1)/(3*Na)+1
iatom=mod(i-1,3*Na)/3+1
ixyz=mod(i-1,3)+1
jcell=(j-1)/(3*Na)+1
jatom=mod(j-1,3*Na)/3+1
jxyz=mod(j-1,3)+1

xcell=mod(jcell-icell+Nx,Nx)+1 !if jcell>icell, xcell=jcell-icell+1, otherwise, xcell=jcell-icell+1+Nx
  
   do l=-N_cut(1),N_cut(1)
   do m=-N_cut(2),N_cut(2)
!   kL11(:,:,i,j)=f(ixyz,jxyz,iatom,jatom,xcell,:,:)
     ycell=mod(l+Ny,Ny)+1
     zcell=mod(m+Nz,Nz)+1
      kR11(l,m,i,j)=f(ixyz,jxyz,iatom,jatom,xcell,ycell,zcell)
 
     if(jcell.le.icell) then
      kR01(l,m,i,j)=f(ixyz,jxyz,iatom,jatom,jcell+ncell-icell+1,ycell,zcell)
     else
      kR01(l,m,i,j)=0d0
     endif
   enddo
   enddo

enddo
enddo

open(301,file="Kmatrix.dat",recl=1000)

write(301,*)"kR11"
do l=-N_cut(1),N_cut(1)
do m=-N_cut(2),N_cut(2)
do i=1,MR
do j=1,MR
write(301,*)l,m,i,j,KR11(l,m,i,j) 
enddo;enddo;enddo;enddo

write(301,*)"kR01"
do l=-N_cut(1),N_cut(1)
do m=-N_cut(2),N_cut(2)
do i=1,MR
do j=1,MR
write(301,*)l,m,i,j,KR01(l,m,i,j)
enddo;enddo;enddo;enddo


end subroutine





subroutine read_QE(Na,Nx,Ny,Nz,f)
implicit none
integer :: Na !number of atoms in unit cell
integer :: Nx,Ny,Nz !The position index in x,y,x direction
double precision,allocatable :: f(:,:,:,:,:,:,:)
!(polarization index (xyz): polarization index : atom index : atom index : unit cell position index i:j:k)
!example: f(1,2,3,4,6,7,8) force between x of third atom and y of fourth atom seperated by vector (6,7,8)
!where R=(6-1)a_1+(7-1)a_2+(8-1)a_3. (1,1,1)is thr origin. a_1,a_2,a_3 are the bravis lattice vector.

integer :: ix,iy,iz,i1,i2,j1,j2,k1,k2,k3
character :: tmpc
double precision :: ftmp
integer :: check_square
double precision :: kk1,kk2,kk3
double precision :: v(3,3),h(3,3)
open(201,file='graphene_1L_PBE_van.fc')

read(201,*)i1,Na
do while(tmpc.ne. 'F')
read(201,'(A2)')tmpc
enddo

read(201,*)Nx,Ny,Nz
write(*,*)"Na,Nx,Ny,Nz=",Na,Nx,Ny,Nz
allocate(f(3,3,Na,Na,Nx,Ny,Nz))

do i1=1,3 !xyz index
do i2=1,3
do j1=1,Na !atom index
do j2=1,Na
read(201,*)tmpc
do k1=1,Nx
do k2=1,Ny
do k3=1,Nz
read(201,*)ix,iy,iz,ftmp
f(i1,i2,j1,j2,ix,iy,iz)=ftmp
!if(k1.eq.ix.and.k2.eq.iy.and.k3.eq.iz)write(*,*)k1,k2,k3,"Yes"
enddo
enddo
enddo

enddo!j2
enddo!j1
enddo!i2
enddo!i1
kk1=1.3d0
kk2=kk1/4d0
kk3=kk1/4d0
check_square=0
if(check_square.eq.1)then
f=0;

do j1=1,Na
f(1,1,j1,j1,1,1,1)=2*kk1+10*kk2
f(2,2,j1,j1,1,1,1)=2*kk1+10*kk2
f(3,3,j1,j1,1,1,1)=8*kk2+4*kk3
enddo

do i1=1,3
do i2=1,3
h(i1,i2)=-kk2
v(i1,i2)=-kk2
h(3,3)=-kk3
v(3,3)=-kk3
enddo
enddo
h(1,1)=-kk1
v(2,2)=-kk1

f(:,:,1,2,1,1,1)=h
f(:,:,2,1,1,1,1)=h
f(:,:,4,3,1,1,1)=h
f(:,:,3,4,1,1,1)=h
f(:,:,2,1,2,1,1)=h
f(:,:,4,3,2,1,1)=h
f(:,:,1,2,8,1,1)=h
f(:,:,3,4,8,1,1)=h

f(:,:,1,3,1,1,1)=v
f(:,:,3,1,1,1,1)=v
f(:,:,4,2,1,1,1)=v
f(:,:,2,4,1,1,1)=v
f(:,:,2,4,1,2,1)=v
f(:,:,1,3,1,2,1)=v
f(:,:,4,2,1,6,1)=v
f(:,:,3,1,1,6,1)=v



endif
end subroutine

end program

