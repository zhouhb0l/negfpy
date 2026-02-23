!see JS Wang, J Wang, JT Lu's review article on quantum thermal transport
!written by Jingtao Lu 
!
!subroutine compatible with wangjian's implementation
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
!local variables
double complex, dimension(:,:), allocatable :: MatA,MatB,Identy
double complex,dimension(:), allocatable :: alpha,beta,work
double complex,dimension(:,:), allocatable :: vl,vr
double precision,allocatable :: rwork(:)
double complex,dimension(:,:),allocatable :: transE,transEInv,Gam
double complex,dimension(:,:),allocatable :: GFInv,TH00,TH11
integer :: lwork,info
integer :: n00,n11,n01,n002  ! ,nshape(2),i,j
integer nnum
integer inverror
character :: jobvl,jobvr
!
!Make sure that H00,H11,H01 are the square matricies and of the same size
call IsSquare(H00,n00)
call IsSquare(H11,n11)
call IsSquare(H01,n01)
if(n00.ne.n01.or.n00.ne.n11)then
  print *, "H00 H11, and H01 are not the same size"
  stop
else
!  print *, "matrix size: ", n00," x ",n00
endif
!
!initialize the matrix 
lwork = n00*8
n002 = n00*2
allocate(MatA(n002,n002),MatB(n002,n002),Identy(n00,n00))
allocate(alpha(n002),beta(n002),work(lwork),vl(n002,n002))
allocate(transE(n00,n00),transEInv(n00,n00),vr(n002,n002))
allocate(Gam(n00,n00),GFInv(n00,n00),rwork(n002*8))
allocate(TH00(n00,n00),TH11(n00,n00))
call Identify(Identy)

TH00 = omega*omega*Identy - H00
TH11 = omega*omega*Identy - H11
MatA(1:n00,1:n00) = TH11
MatA(n00+1:n002,1:n00) = conjg(transpose(H01))
MatA(1:n00,n00+1:n002) = -Identy(:,:)
MatA(n00+1:n002,n00+1:n002)=cmplx(0.d0,0.d0)

MatB(1:n00,1:n00) = H01
MatB(n00+1:n002,1:n00) = cmplx(0.d0,0.d0)
MatB(1:n00,n00+1:n002) = cmplx(0.d0,0.d0)
MatB(n00+1:n002,n00+1:n002)=Identy(:,:)
!
!solve the generalized eigen value problem
JOBVL = 'N'
JOBVR = 'V'
call zggev(jobvl,jobvr,n002,matA,n002,MatB,n002,&
        alpha,beta,vl,n002,vr,n002,work,lwork,rwork,info)
if(info.ne.0)then
  stop "zggev return value error"
endif
!
!find eigen values those absolute values less than 1
call eigenvalue(vr,alpha,beta,transE,gam,nnum)
!pseudo inverse
call zpinverse(transE(1:n00,1:nnum), transEInv(1:nnum,1:n00), inverror)
!call zsquare_matrix_inverse(transE, transEInv, inverror)
if(inverror.ne.0)then
  stop "failed to find the matrix pseudo inverse"
endif
!
!find g11
GFInv(:,:) = TH11(:,:) - matmul(matmul(H01(:,:),transE(:,:)),&
        matmul(gam(:,:),transEInv(:,:)))
call zsquare_matrix_inverse(GFInv, GF, inverror)
if(inverror.ne.0)then
  stop "matrix inversion error."
endif
!
!find g00
GFInv(:,:) = TH00(:,:)-matmul(matmul(H01(:,:),GF(:,:)),conjg(transpose(H01(:,:))))
call zsquare_matrix_inverse(GFInv, GF, inverror)
if(inverror.ne.0)then
  stop "matrix inversion error."
endif
!
deallocate(MatA,MatB,Identy)
deallocate(alpha,beta,vl,vr,work,rwork)
deallocate(transE,transEInv,Gam)
deallocate(GFInv,TH00,TH11)

contains 
!
subroutine eigenvalue(vr,alpha,beta,transE,gam,nnum)
!find eigen values that are smaller than 1
!and construct matrix \Gamma  and E
!
implicit none
double complex,dimension(:,:),intent(in) :: vr
double complex,dimension(:),intent(inout) :: alpha
double complex,dimension(:),intent(in) :: beta
double complex,dimension(:,:),intent(out) :: transE,gam
integer,intent(out) :: nnum
integer nsize,i,nt,ng,nv
!
!matrix size check
call IsSquare(transE,nt)
call IsSquare(gam,ng)
call IsSquare(vr,nv)
if(nt.ne.ng)stop "transE and gam have different sizes. This can't happen."
if(nt.ne.nv/2)stop "transE and vr shape inconsistent. This can't happen."
nsize = size(alpha)
if(nv.ne.nsize)stop "alpha, vr shape inconsistent. This can't happen."
nsize = size(beta)
if(nv.ne.nsize)stop "beta, vr shape inconsistent. This can't happen."
!
!
gam(:,:) = cmplx(0.d0,0.d0);transE(:,:) = cmplx(0.d0,0.d0)
nnum = 0
do i=1,nsize
if(abs(beta(i)) .eq. 0.d0)then
!if beta(i) is zero, the eigen value is infinite. We just set it to a number
!greater than 1.
!  print *, "The eigen value has infinity."
  alpha(i) = 10.d0
else
 alpha(i) = alpha(i)/beta(i)
!  if(abs(alpha(i)).lt.1.and.abs(alpha(i)).ne.0.)then
 if(abs(alpha(i)).lt.1.0) then
   nnum = nnum + 1
   if(nnum.gt.nt)then
     stop "nnum should not be such large"
   endif
   gam(nnum,nnum) = alpha(i)
   transE(1:nt,nnum) = vr(1:nt,i)
 endif
endif
enddo
!
end subroutine eigenvalue
!
subroutine IsSquare(H,Nsize)
!judge if matrix H is square, return its size
implicit none
double complex, dimension(:,:),intent(in) :: H
integer :: Nsize

integer :: nshape(2)

nshape(:) = shape(H(:,:))
if(nshape(1).ne.nshape(2))then
  print *, "Not a square matrix!"
  stop
else
  nsize = nshape(1)
endif
end subroutine IsSquare
!
subroutine Identify(zmatrix)
!Generate a square identity matrix
implicit none
double complex, intent(inout) :: zmatrix(:,:)

!local variables       
integer::  num_dim(1:2)
integer :: i

num_dim(:) = shape(zmatrix)
if(num_dim(1).ne.num_dim(2))then
stop "Not a square matrix in Identify."
endif

zmatrix(:,:) = cmplx(0.,0.)     
do i=1,num_dim(1)
zmatrix(i,i)=cmplx(1.d0,0.)
enddo
end subroutine
!
subroutine zsquare_matrix_inverse(a_in, a_out, error_id)
!copy from NEGF source code
double complex, intent(in)::   a_in(:,:)
double complex, intent(out)::  a_out(:,:)
integer, intent(out)::  error_id

! Local variables
integer::  nn,  Lda,  Lwork  !, info
integer::  num_dim(1:2), row_num, col_num
integer, allocatable::         ipiv(:)
double complex, allocatable::  work(:), temp_a(:,:)
integer:: info_1, info_2

a_out(:,:) = cmplx(0.0, 0.0)
error_id = - 1000

num_dim(:) = shape(a_in)
row_num = num_dim(1); col_num = num_dim(2)

if(row_num.eq.1.and.col_num.eq.1)then
 if(abs(a_in(1,1)) .eq. 0.)then
   pause "This matrix is 1x1. But it is zero."
 endif
 a_out = 1./a_in
 error_id = 0
else
allocate(ipiv(1:row_num), work(1:row_num), &
           temp_a(1:row_num, 1:col_num) )

temp_a(:,:) = a_in(:,:) 

if( row_num /= col_num ) then
 print*, 'error in matrix inversion, shape error'
 pause 'Press Ctr-C to break this program'
end if

ipiv(:) = 0
nn = row_num;   Lda = row_num;    Lwork = row_num

CALL zgetrf(nn, nn, temp_a, Lda,  ipiv, info_1 )

a_out(:,:) = temp_a(:,:)
CALL zgetri(nn, a_out, Lda, ipiv,  work,  Lwork,  info_2 )

error_id = abs(info_1) + abs(info_2 )

deallocate(ipiv, work, temp_a)
endif

end subroutine zsquare_matrix_inverse
!


subroutine zpinverse(matin,matout,error)
!find the pseudo inverse using the SVD method by calling LAPACK routines
!
!LAPACK ROUTINE USED:
!ZGESVD DLAMCH
!
implicit none
double complex,intent(in) :: matin(:,:)
double complex,intent(out) :: matout(:,:)
integer,intent(out) :: error
!
CHARACTER JOBU, JOBVT
INTEGER   LWORK, drwork,ds
DOUBLE PRECISION, allocatable :: RWORK(:), S(:)
double COMPLEX,allocatable ::    mata(:,:),U(:,:), VT(:,:), WORK(:)
!
integer nshape(2),nx,ny,i,j,k
double precision dfmin,deps,dlamch
!find machine parameters
deps = dlamch('P')
dfmin = dlamch('S')
!
jobu = 'A'
jobvt = 'A'
nshape(:) = shape(matin(:,:))
nx = nshape(1); ny = nshape(2)
!drwork = max(3*min(nx,ny),5*min(nx,ny)-4)
drwork = 5*min(nx,ny)
ds = min(nx,ny)
lwork = (2*ds+MAX(nx,ny))*2
allocate(mata(nx,ny),rwork(drwork))
allocate(s(ds),u(nx,nx),vt(ny,ny),work(lwork))
mata = matin
call ZGESVD(JOBU, JOBVT, nx, ny, mata, nx, S, U, nx, VT,&
                        ny, WORK, LWORK, RWORK, error)
do i=1,ny
do j=1,nx
matout(i,j)=cmplx(0.d0,0.d0)
do k=1,ds
if(S(k).ge.max(dfmin,S(1)*deps))then
matout(i,j) = matout(i,j) + conjg(vt(k,i))*conjg(u(j,k))/s(k)
endif
enddo
enddo
enddo
deallocate(mata,rwork,s,u,vt,work)
end subroutine zpinverse


end subroutine sfg_solve


! need in main
! <<<<<  如果是左边的 surface green function 需要倒转一下    >>>>>>
subroutine  reverse_green(input_matrix)
   implicit none
   double complex, intent(inout):: input_matrix(:,:)

   ! Local variables
   double complex, dimension(:,:), allocatable::   tmp_matrix
   integer::  num_shape(1:2),  num_row,  num_col
   integer::  i, j, ii, jj, error_id

   num_shape(:) = shape(input_matrix)

   num_row = num_shape(1);  num_col = num_shape(2)
   if( num_row .ne. num_col) then
     print *, 'shape error in surface green function in reverse_green'
     print *, 'row=', num_row, ' col=', num_col
     stop
   end if

   allocate(tmp_matrix(1:num_row, 1:num_col), stat = error_id )
   if(error_id .ne. 0 )then
     print*, 'memory allocation error in reverse_green'
     stop
   end if

   do i=1, num_row
      do j=1, num_row
         ii = num_row - i + 1
	 jj = num_row - j + 1
	 tmp_matrix(ii,jj) = input_matrix(i,j)
      end do ! j
   end do ! i
 
   input_matrix(:,:) = tmp_matrix(:,:)

   deallocate(tmp_matrix)
    
end subroutine 
