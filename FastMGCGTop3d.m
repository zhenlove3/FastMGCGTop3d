%%%% Fast 3D TOPOLOGY OPTIMIZATION CODE, MGCG ANALYSIS %%%%
% This code is freely available 
% Please reference the article: 
% Efficient acceleration strategies for multigrid preconditioned conjugate gradients in fast 3D topology optimization
% run:
% tic;FastMGCGTop3d(48,24,24,0.12,3.0,sqrt(3),1,4,0.1,100);toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Disclaimer:                                                              %
% The authors reserve all rights but do not guarantee that the code is     %
% free from errors. Furthermore, the authors shall not be liable in any    %
% event caused by the use of the program.                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function FastMGCGTop3d(nelx,nely,nelz,volfrac,penal,rmin,ft,nl,cgtol,cgmax)
    %% MATERIAL PROPERTIES
    E0 = 1;
    Emin = 1e-9;
    nu = 0.3;
    
    %% PREPARE FINITE ELEMENT ANALYSIS
    % KE = Ke3D(nu, sqrt(3)/3);   % FEM
    KE = Ke3D(nu, 1);           % FDM
    % SDC-part of KE
    % SKE((1:8)*3  , (1:8)*3  ) = KE((1:8)*3  , (1:8)*3  );
    % SKE((1:8)*3-1, (1:8)*3-1) = KE((1:8)*3-1, (1:8)*3-1);
    % SKE((1:8)*3-2, (1:8)*3-2) = KE((1:8)*3-2, (1:8)*3-2);  

    % Prepare fine grid
    nelem = nelx*nely*nelz;
    nx = nelx+1; ny = nely+1; nz = nelz+1;
    ndof = 3*nx*ny*nz;
    
    % Define loads and supports (cantilever)
    nodenrs(1:ny,1:nz,1:nx) = reshape(1:ny*nz*nx,ny,nz,nx);
    F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,-sin((0:nely)/nely*pi),ndof,1); % Sine load, bottom right
    U = zeros(ndof,1);
    %% PREPARE FILTER
    cr = 1-ceil(rmin) : ceil(rmin)-1;
    [dy, dx, dz] = meshgrid(cr,cr,cr);
    h = max(0,rmin-sqrt(dx.^2+dy.^2+dz.^2));
    Hs1 = convn(ones(nely,nelz,nelx),h,'same');
    %% INITIALIZE ITERATION
    x = volfrac*ones(nelem,1);
    xPhys = x;
    loop = 0;
    change = 1;

    edofMat2 = cell(1,nl);
    iK2 = cell(1,nl);
    jK2 = cell(1,nl);
    fixeddofs2 = cell(1,nl);
    Pu2 = cell(1,nl);
    Null2 = cell(1,nl);
    Bull2 = cell(1,nl);
    rho = cell(1,nl);
    invD = cell(1,nl-1);
    nel = [nelx,nely,nelz];
    for l = 1:nl
        [edofMat2{l}, iK2{l}, jK2{l}, fixeddofs2{l}, Pu2{l}, Null2{l}, Bull2{l}] = subphi(nel/(2^(l-1)));
    end
    % Prologation operators
    Pu = cell(1,nl-1);
    for l = 1:nl-1
        [Pu{l}] = prepcoarse(nelz/2^(l-1),nely/2^(l-1),nelx/2^(l-1));
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
    %% Coefficients  of matrix vector multiplication 
    cfs = cell(1,nl);
    for l = 1:nl
        cfs{l}.fixeddofs2 = fixeddofs2{l};
        cfs{l}.hsize = [1, 1, 1];
        cfs{l}.meshsize = nel/(2^(l-1));
        cfs{l}.dofsize = cfs{l}.meshsize + 1;
        cfs{l}.alldof = 3*prod(cfs{l}.dofsize);
        cfs{l}.kc = 2^(l-1);
        cfs{l}.l = l;
    end

    att = (1-nu)/((1+nu)*(1-2*nu));
    btt = nu/((1+nu)*(1-2*nu));
    ctt = (0.5 -nu)/((1+nu)*(1-2*nu)); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
    %% START ITERATION
    maxloop= 50;
    ALLcgiters = 0;
    lmid = 1e10;
    while change > 1e-2 && loop < maxloop
        loop = loop+1;
        rho{1} = Emin+xPhys.^penal*(E0-Emin);
        for l = 2:nl
            rho{l} = Pu2{l}'*rho{l-1}/8;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        for l = 1:nl-1
            rhox = reshape(rho{l}, cfs{l}.meshsize([2,3,1]));
            rhox = permute(rhox, [3,1,2]);
            bb1 = rhox* att;
            bbb = rhox* btt;
            ccc = rhox* ctt;
            cfs{l} = Ax_cf(cfs{l}, bb1, ccc, ccc, bbb, ccc);
            cfs{l}.loop = loop;
        end
        %% diag(K_h)^{-1}
        for l = 1:nl-1
            u111 = ones(cfs{l}.alldof, 1); 
            tmp = Audiag(cfs{l}, u111);
            invD{l}= 1./tmp;
        end

        l = nl;
        KH = Null2{l}'*sparse(iK2{l},jK2{l},KE(:)*rho{l}')*Null2{l} + Bull2{l};
        KH =  KH*(2^(l-1));
        Lfac = chol(KH,'lower'); Ufac = Lfac';

        if (mod(loop, 50) == 0)
            cgtol = min(cgtol, 1e-6);
        end
        [cgiters,cgres,U] = mgcg(cfs, invD, F,U,Lfac,Ufac,Pu,nl,cgtol,cgmax);
        % toc
        %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce = sum(U(edofMat2{1})*KE.*U(edofMat2{1}),2);
        c = U'*F;
        dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
        dv = ones(nelem(1),1);
        %% FILTERING/MODIFICATION OF SENSITIVITIES
        if ft == 1
            xdc = x(:).*dc(:);
            tmp = convn(reshape(xdc, nely,nelz,nelx), h, 'same')./Hs1;
            dc(:) = tmp(:)./max(1e-3,x(:));
        end
        %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
        g = mean(xPhys(:)) - volfrac;
        l1 = 0; l2 = lmid*2; move = 0.2;
        while (l2-l1)/(l1+l2) > 1e-6
            lmid = 0.5*(l2+l1);
            xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
            gt = g + sum((dv(:).*(xnew(:)-x(:))));
            if gt > 0, l1 = lmid; else l2 = lmid; end
        end
        change = max(abs(xnew(:)-x(:)));
        x = xnew;  xPhys = xnew;
        %% PRINT RESULTS
        fprintf(' It.:%4i Obj.:%6.3e Vol.:%6.3e ch.:%4.2e relres: %4.2e iters: %4i \n',...
            loop,c,mean(xPhys(:)),change,cgres,cgiters);
        ALLcgiters = ALLcgiters + cgiters;
    end
    disp(ALLcgiters);
    %% PLOT
    isovals = shiftdim(reshape(xPhys,nely,nelz,nelx),2);
    isovals = smooth3(isovals,'box',1);
    patch(isosurface(isovals,0.8),'FaceColor',[0 0 1],'EdgeColor','none');
    patch(isocaps(isovals,0.8),'FaceColor',[1 0 0],'EdgeColor','none');
    view(3); axis equal tight off; camlight;
end


function v = Au(cf, u)
    % v = Ausdc(cf, u);
    v = Au2(cf, u);
end

function v = Au2(cf, u)
    fixu = u(cf.fixeddofs2);
    u(cf.fixeddofs2) = 0;
    u1b = reshape(u(1:3:end), cf.dofsize([2,3,1]));
    u2b = reshape(u(2:3:end), cf.dofsize([2,3,1]));
    u3b = reshape(u(3:3:end), cf.dofsize([2,3,1]));
    u1b = permute(u1b, [3,1,2]);
    u2b = permute(u2b, [3,1,2]);
    u3b = permute(u3b, [3,1,2]);
    u1b = reshape(u1b(:), cf.dofsize);
    u2b = reshape(u2b(:), cf.dofsize);
    u3b = reshape(u3b(:), cf.dofsize);
    [t1, t2, t3] = Ax3D(cf, u1b, u2b, u3b);
    t1 = permute(t1, [2,3,1]);
    t2 = permute(t2, [2,3,1]);
    t3 = permute(t3, [2,3,1]);
    v = [t1(:), t2(:),t3(:)]';
    v = v(:);
    v(cf.fixeddofs2) = fixu ;
    v = v*cf.kc;
end

function v = Ausdc(cf, u)
    fixu = u(cf.fixeddofs2);
    u(cf.fixeddofs2) = 0;
    u1b = reshape(u(1:3:end), cf.dofsize([2,3,1]));
    u2b = reshape(u(2:3:end), cf.dofsize([2,3,1]));
    u3b = reshape(u(3:3:end), cf.dofsize([2,3,1]));
    u1b = permute(u1b, [3,1,2]);
    u2b = permute(u2b, [3,1,2]);
    u3b = permute(u3b, [3,1,2]);
    u1b = reshape(u1b(:), cf.dofsize);
    u2b = reshape(u2b(:), cf.dofsize);
    u3b = reshape(u3b(:), cf.dofsize);
    [t1, t2, t3] = Ax3Dsdc(cf, u1b, u2b, u3b);
    t1 = permute(t1, [2,3,1]);
    t2 = permute(t2, [2,3,1]);
    t3 = permute(t3, [2,3,1]);
    v = [t1(:), t2(:),t3(:)]';
    v = v(:);
    v(cf.fixeddofs2) = fixu ;
    v = v*cf.kc;
end

function v = Audiag(cf, u)
    fixu = u(cf.fixeddofs2);
    u(cf.fixeddofs2) = 0;
    u1b = reshape(u(1:3:end), cf.dofsize([2,3,1]));
    u2b = reshape(u(2:3:end), cf.dofsize([2,3,1]));
    u3b = reshape(u(3:3:end), cf.dofsize([2,3,1]));
    u1b = permute(u1b, [3,1,2]);
    u2b = permute(u2b, [3,1,2]);
    u3b = permute(u3b, [3,1,2]);
    u1b = reshape(u1b(:), cf.dofsize);
    u2b = reshape(u2b(:), cf.dofsize);
    u3b = reshape(u3b(:), cf.dofsize);
    [t1, t2, t3] = Ax3Ddiag(cf, u1b, u2b, u3b);
    t1 = permute(t1, [2,3,1]);
    t2 = permute(t2, [2,3,1]);
    t3 = permute(t3, [2,3,1]);
    v = [t1(:), t2(:),t3(:)]';
    v = v(:);
    v(cf.fixeddofs2) = fixu ;
    v = v*cf.kc;
end


%% FUNCTION mgcg - MULTIGRID PRECONDITIONED CONJUGATE GRADIENTS
function [i,relres,u] = mgcg(cfs, invD, b,u,Lfac,Ufac,Pu,nl,tol,maxiter)
    cf = cfs{1};
    % r = b - A0*u;
    r = b - Au2(cf, u);
    res0 = norm(b); 
    % Jacobi smoother
    omega = 0.6;
    for i = 1:1e6 
        z = VCycle(cfs,r,Lfac,Ufac,Pu,1,nl,invD,omega);
        rz = r'*z;
        if i == 1
            p = z;
        else
            beta = rz/rz_p;
            p = beta*p + z;
        end
        % q = A0*p;
        q = Au2(cf, p);
        dpr = p'*q;
        alpha = rz/dpr;
        u = u+alpha*p;
        r = r-alpha*q;
        rz_p = rz;
        relres = norm(r)/res0;
        if  (relres < tol || i>=maxiter) 
            break
        end
    end
end


%% FUNCTION VCycle - COARSE GRID CORRECTION
function z = VCycle(cfs,r,Lfac,Ufac,Pu,l,nl,invD,omega)
    if (l == nl)
        z = Ufac \ (Lfac \ r);
    else 
        %%% N-cycle
        z = Pu{l}*VCycle(cfs, Pu{l}'*r,Lfac,Ufac,Pu,l+1,nl,invD,omega);
        z = z + omega*invD{l}.*(r-Au(cfs{l}, z));
        z = z + Pu{l}*VCycle(cfs, Pu{l}'*(r - Au(cfs{l}, z)),Lfac,Ufac,Pu,l+1,nl,invD,omega);
        z = z + omega*invD{l}.*(r-Au(cfs{l}, z));

        %%% W-cycle
        % z = omega*invD{l}.*r;
        % z = z + Pu{l}*VCycle(cfs,Pu{l}'*(r - Au(cfs{l}, z)),Lfac,Ufac,Pu,l+1,nl,invD,omega);
        % z = z + omega*invD{l}.*(r-Au(cfs{l}, z));
        % z = z + Pu{l}*VCycle(cfs,Pu{l}'*(r - Au(cfs{l}, z)),Lfac,Ufac,Pu,l+1,nl,invD,omega);
        % z = z + omega*invD{l}.*(r-Au(cfs{l}, z));

        %% V-cycle
        % z = omega*invD{l}.*r;
        % z = z + Pu{l}*VCycle(cfs,Pu{l}'*(r - Au(cfs{l}, z)),Lfac,Ufac,Pu,l+1,nl,invD,omega);
        % z = z + omega*invD{l}.*(r-Au(cfs{l}, z));

    end
end

%% FUNCTION Ke3D - ELEMENT STIFFNESS MATRIX
function KE = Ke3D(nu, sc)
    h = 0.5-nu;
    D=[1-nu  nu   nu 0 0 0;
        nu 1-nu   nu 0 0 0;
        nu   nu 1-nu 0 0 0;
        0     0    0 h 0 0;
        0     0    0 0 h 0;
        0     0    0 0 0 h];
    xs = sc*[-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; 
             -1 -1  1; 1 -1  1; 1 1  1; -1 1  1];
    w =  ones( 8,  1);  z = zeros( 8,  1); B = zeros( 6, 24);
    KE= zeros(24, 24);  dN= zeros( 8,  3);
    for q = 1:8
        r=xs(q,1); s=xs(q,2); t=xs(q,3);
        dN(1, :) = [-(1-s)*(1-t), -(1-r)*(1-t), -(1-r)*(1-s)]/8;
        dN(2, :) = [ (1-s)*(1-t), -(1+r)*(1-t), -(1+r)*(1-s)]/8;
        dN(3, :) = [ (1+s)*(1-t),  (1+r)*(1-t), -(1+r)*(1+s)]/8;
        dN(4, :) = [-(1+s)*(1-t),  (1-r)*(1-t), -(1-r)*(1+s)]/8;
        dN(5, :) = [-(1-s)*(1+t), -(1-r)*(1+t),  (1-r)*(1-s)]/8;
        dN(6, :) = [ (1-s)*(1+t), -(1+r)*(1+t),  (1+r)*(1-s)]/8;
        dN(7, :) = [ (1+s)*(1+t),  (1+r)*(1+t),  (1+r)*(1+s)]/8;
        dN(8, :) = [-(1+s)*(1+t),  (1-r)*(1+t),  (1-r)*(1+s)]/8;
        N1 = dN(:, 1); N2 = dN(:, 2); N3 = dN(:, 3);
        B(1, :) = reshape([N1 z  z].', 1, []);
        B(2, :) = reshape([z N2  z].', 1, []);
        B(3, :) = reshape([z z  N3].', 1, []);
        B(4, :) = reshape([N2 N1 z].', 1, []);
        B(5, :) = reshape([z N3 N2].', 1, []);
        B(6, :) = reshape([N3 z N1].', 1, []);
        KE = KE + B'*D*B*w(q);
    end
    KE = KE/(2*(1+nu)*(1-2*nu));
end


%% FUNCTION prepcoarse - PREPARE MG PROLONGATION OPERATOR
function [Pu] = prepcoarse(nex,ney,nez)
% Assemble state variable prolongation
maxnum = nex*ney*nez*20;
iP = zeros(maxnum,1); jP = zeros(maxnum,1); sP = zeros(maxnum,1);
nexc = nex/2; neyc = ney/2; nezc = nez/2;
% Weights for fixed distances to neighbors on a structured grid 
vals = [1,0.5,0.25,0.125];
cc = 0;
for nx = 1:nexc+1
    for ny = 1:neyc+1
        for nz = 1:nezc+1
            col = (nx-1)*(neyc+1)+ny+(nz-1)*(neyc+1)*(nexc+1); 
            % Coordinate on fine grid
            nx1 = nx*2 - 1; ny1 = ny*2 - 1; nz1 = nz*2 - 1;
            % Loop over fine nodes within the rectangular domain
            for k = max(nx1-1,1):min(nx1+1,nex+1)
                for l = max(ny1-1,1):min(ny1+1,ney+1)
                    for h = max(nz1-1,1):min(nz1+1,nez+1)
                        row = (k-1)*(ney+1)+l+(h-1)*(nex+1)*(ney+1); 
                        % Based on squared dist assign weights: 1.0 0.5 0.25 0.125
                        ind = 1+((nx1-k)^2+(ny1-l)^2+(nz1-h)^2);
                        cc=cc+1; iP(cc)=3*row-2; jP(cc)=3*col-2; sP(cc)=vals(ind);
                        cc=cc+1; iP(cc)=3*row-1; jP(cc)=3*col-1; sP(cc)=vals(ind);
                        cc=cc+1; iP(cc)=3*row; jP(cc)=3*col; sP(cc)=vals(ind);
                    end
                end
            end
        end
    end
end
% Assemble matrices
Pu = sparse(iP(1:cc),jP(1:cc),sP(1:cc));
end

function [Pu] = prepcoarse2(nex,ney,nez)
    % Assemble state variable prolongation
    maxnum = nex*ney*nez;
    iP = zeros(maxnum,1); jP = zeros(maxnum,1); sP = zeros(maxnum,1);
    nexc = nex/2; neyc = ney/2; nezc = nez/2;
    cc = 0;
    for nx = 1:nexc
        for ny = 1:neyc
            for nz = 1:nezc
                col = (nx-1)*(neyc)+ny+(nz-1)*(neyc)*(nexc); 
                % Coordinate on fine grid
                nx1 = nx*2-1; ny1 = ny*2-1; nz1 = nz*2-1;
                % Loop over fine nodes within the rectangular domain
                for k = nx1:nx1+1
                    for l = ny1:ny1+1
                        for h = nz1:nz1+1
                            row = (k-1)*(ney)+l+(h-1)*(nex)*(ney); 
                            cc=cc+1; iP(cc)=row;   jP(cc)=col;   sP(cc)=1;
                        end
                    end
                end
            end
        end
    end
    % Assemble matrices
    Pu = sparse(iP(1:cc),jP(1:cc),sP(1:cc));
end
function [edofMat2, iK2, jK2, fixeddofs2, Pu2, Null2, Bull2] = subphi(nel2)
    nelx2 = nel2(1); nely2 = nel2(2); nelz2 = nel2(3); 
    % Prepare fine grid
    nelem2= nelx2*nely2*nelz2;
    nx2 = nelx2+1; ny2 = nely2+1; nz2 = nelz2+1;
    ndof2 = 3*nx2*ny2*nz2;
    nodenrs2(1:ny2,1:nz2,1:nx2) = reshape(1:ny2*nz2*nx2,ny2,nz2,nx2);
    edofVec2(1:nelem2,1) = reshape(3*nodenrs2(1:ny2-1,1:nz2-1,1:nx2-1)+1,nelem2,1);
    edofMat2(1:nelem2,1:24) = repmat(edofVec2(1:nelem2),1,24) + ...
        repmat([0 1 2 3*ny2*nz2+[0 1 2 -3 -2 -1] -3 -2 -1 ...
        3*ny2+[0 1 2] 3*ny2*(nz2+1)+[0 1 2 -3 -2 -1] 3*ny2+[-3 -2 -1]],nelem2,1);
    iK2 = reshape(kron(edofMat2(1:nelem2,1:24),ones(24,1))',576*nelem2,1);
    jK2 = reshape(kron(edofMat2(1:nelem2,1:24),ones(1,24))',576*nelem2,1);
    fixeddofs2 = 1:3*(nely2+1)*(nelz2+1);
    N2 = ones(ndof2,1); N2(fixeddofs2) = 0; Null2 = spdiags(N2,0,ndof2,ndof2);
    Bull2 = speye(ndof2,ndof2) - Null2;
    Pu2 = prepcoarse2(nelz2*2, nely2*2, nelx2*2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cf = Ax_cf(cf, bb1, bb2, bb3, bbb, ccc)
    [cex, cx1, cx2] = uxx(bb1, cf.meshsize, cf.hsize);
    [cey, cy1, cy2] = uyy(bb2, cf.meshsize, cf.hsize);
    [cez, cz1, cz2] = uzz(bb3, cf.meshsize, cf.hsize);
    cf.ucx1 = cx1;
    cf.ucx2 = cx2;
    cf.ucy1 = cy1;
    cf.ucy2 = cy2;
    cf.ucz1 = cz1;
    cf.ucz2 = cz2;
    cf.ucen = cex + cey + cez;

    [cex, cx1, cx2] = uxx(bb2, cf.meshsize, cf.hsize);
    [cey, cy1, cy2] = uyy(bb1, cf.meshsize, cf.hsize);
    [cez, cz1, cz2] = uzz(bb3, cf.meshsize, cf.hsize);
    cf.vcx1 = cx1;
    cf.vcx2 = cx2;
    cf.vcy1 = cy1;
    cf.vcy2 = cy2;
    cf.vcz1 = cz1;
    cf.vcz2 = cz2;
    cf.vcen = cex + cey + cez;

    [cex, cx1, cx2] = uxx(bb3, cf.meshsize, cf.hsize);
    [cey, cy1, cy2] = uyy(bb2, cf.meshsize, cf.hsize);
    [cez, cz1, cz2] = uzz(bb1, cf.meshsize, cf.hsize);
    cf.wcx1 = cx1;
    cf.wcx2 = cx2;
    cf.wcy1 = cy1;
    cf.wcy2 = cy2; 
    cf.wcz1 = cz1;
    cf.wcz2 = cz2;
    cf.wcen = cex + cey + cez;

    [ce, xL, xR, yL, yR, co] = uxz(bbb, cf.meshsize, cf.hsize);
    [ce2, xL2, xR2, yL2, yR2, co2] = uzx(ccc, cf.meshsize, cf.hsize);
    cf.cenE13 = ce + ce2;
    cf.cx1E13 = xL + xL2;
    cf.cx2E13 = xR + xR2;
    cf.cz1E13 = yL + yL2;
    cf.cz2E13 = yR + yR2;
    cf.coyE13 = co + co2;

    [ce, xL, xR, yL, yR, co] = uxy(bbb, cf.meshsize, cf.hsize);
    [ce2, xL2, xR2, yL2, yR2, co2] = uyx(ccc, cf.meshsize, cf.hsize);
    cf.cenE12 = ce + ce2;
    cf.cx1E12 = xL + xL2;
    cf.cx2E12 = xR + xR2;
    cf.cy1E12 = yL + yL2;
    cf.cy2E12 = yR + yR2;
    cf.cozE12 = co + co2;

    [ce, xL, xR, yL, yR, co] = uxy(ccc, cf.meshsize, cf.hsize);
    [ce2, xL2, xR2, yL2, yR2, co2] = uyx(bbb, cf.meshsize, cf.hsize);
    cf.cenE21 = ce + ce2;
    cf.cx1E21 = xL + xL2;
    cf.cx2E21 = xR + xR2;
    cf.cy1E21 = yL + yL2;
    cf.cy2E21 = yR + yR2;
    cf.cozE21 = co + co2;

    [ce, xL, xR, yL, yR, co] = uyz(bbb, cf.meshsize, cf.hsize);
    [ce2, xL2, xR2, yL2, yR2, co2] = uzy(ccc, cf.meshsize, cf.hsize);
    cf.cenE23 = ce + ce2;
    cf.cy1E23 = xL + xL2;
    cf.cy2E23 = xR + xR2;
    cf.cz1E23 = yL + yL2;
    cf.cz2E23 = yR + yR2;
    cf.coxE23 = co + co2;

    [ce, xL, xR, yL, yR, co] = uyz(ccc, cf.meshsize, cf.hsize);
    [ce2, xL2, xR2, yL2, yR2, co2] = uzy(bbb, cf.meshsize, cf.hsize);
    cf.cenE32 = ce + ce2;
    cf.cy1E32 = xL + xL2;
    cf.cy2E32 = xR + xR2;
    cf.cz1E32 = yL + yL2;
    cf.cz2E32 = yR + yR2;
    cf.coxE32 = co + co2;

    [ce, xL, xR, yL, yR, co] = uxz(ccc, cf.meshsize, cf.hsize);
    [ce2, xL2, xR2, yL2, yR2, co2] = uzx(bbb, cf.meshsize, cf.hsize);
    cf.cenE31 = ce + ce2;
    cf.cx1E31 = xL + xL2;
    cf.cx2E31 = xR + xR2;
    cf.cz1E31 = yL + yL2;
    cf.cz2E31 = yR + yR2;
    cf.coyE31 = co + co2;
end

function [Ax,Bx,Cx] = Ax3D(cf, x1, x2, x3)
    x2 = -x2;
    Ax = cf.ucen.*x1 + cf.cenE13.*x3 + cf.cenE12.*x2;

    Ax(2:end  , :, :) = Ax(2:end,   :, :) - x1(1:end-1, :, :) .* cf.ucx1 - x3(1:end-1, :, :) .* cf.cx1E13 - x2(1:end-1, :, :) .* cf.cx1E12;
    Ax(1:end-1, :, :) = Ax(1:end-1, :, :) - x1(2:end, :,   :) .* cf.ucx2 - x3(2:end, :,   :) .* cf.cx2E13 - x2(2:end, :,   :) .* cf.cx2E12;
    Ax(:, 2:end,   :) = Ax(:, 2:end,   :) - x1(:, 1:end-1, :) .* cf.ucy1 - x2(:, 1:end-1, :) .* cf.cy1E12;
    Ax(:, 1:end-1, :) = Ax(:, 1:end-1, :) - x1(:, 2:end,   :) .* cf.ucy2 - x2(:, 2:end,   :) .* cf.cy2E12;
    Ax(:, :, 2:end  ) = Ax(:, :, 2:end  ) - x1(:, :, 1:end-1) .* cf.ucz1 - x3(:, :, 1:end-1) .* cf.cz1E13;
    Ax(:, :, 1:end-1) = Ax(:, :, 1:end-1) - x1(:, :, 2:end  ) .* cf.ucz2 - x3(:, :, 2:end  ) .* cf.cz2E13;

    Ax(2:end,  :,1:end-1) = Ax(2:end,  :,1:end-1) + cf.coyE13 .* x3(1:end-1,:,2:end);
    Ax(1:end-1,:,2:end)   = Ax(1:end-1,:,2:end)   + cf.coyE13 .* x3(2:end,  :,1:end-1);
    Ax(2:end  ,:,2:end)   = Ax(2:end  ,:,2:end)   - cf.coyE13 .* x3(1:end-1,:,1:end-1);
    Ax(1:end-1,:,1:end-1) = Ax(1:end-1,:,1:end-1) - cf.coyE13 .* x3(2:end,  :,2:end);

    Ax(2:end, 1:end-1, :) = Ax(2:end, 1:end-1, :) + cf.cozE12.*x2(1:end-1, 2:end, :);
    Ax(1:end-1, 2:end, :) = Ax(1:end-1, 2:end, :) + cf.cozE12.*x2(2:end, 1:end-1, :);
    Ax(2:end, 2:end, :)   = Ax(2:end, 2:end, :)   - cf.cozE12.*x2(1:end-1, 1:end-1,:);
    Ax(1:end-1,1:end-1,:) = Ax(1:end-1, 1:end-1,:)- cf.cozE12.*x2(2:end, 2:end, :);

    %%%%%%%%%%
    Bx = cf.vcen.*x2 + cf.cenE21.*x1 + cf.cenE23.*x3;
    Bx(2:end  , :, :) = Bx(2:end,   :, :) - x2(1:end-1, :, :) .* cf.vcx1 - x1(1:end-1, :, :) .* cf.cx1E21;
    Bx(1:end-1, :, :) = Bx(1:end-1, :, :) - x2(2:end, :,   :) .* cf.vcx2 - x1(2:end, :,   :) .* cf.cx2E21;
    Bx(:, 2:end,   :) = Bx(:, 2:end,   :) - x2(:, 1:end-1, :) .* cf.vcy1 - x1(:, 1:end-1, :) .* cf.cy1E21 - x3(:, 1:end-1, :) .* cf.cy1E23;
    Bx(:, 1:end-1, :) = Bx(:, 1:end-1, :) - x2(:, 2:end,   :) .* cf.vcy2 - x1(:, 2:end,   :) .* cf.cy2E21 - x3(:, 2:end,   :) .* cf.cy2E23;
    Bx(:, :, 2:end  ) = Bx(:, :, 2:end  ) - x2(:, :, 1:end-1) .* cf.vcz1 - x3(:, :, 1:end-1) .* cf.cz1E23;
    Bx(:, :, 1:end-1) = Bx(:, :, 1:end-1) - x2(:, :, 2:end  ) .* cf.vcz2 - x3(:, :, 2:end  ) .* cf.cz2E23;

    Bx(2:end, 1:end-1, :) = Bx(2:end, 1:end-1, :) + cf.cozE21.*x1(1:end-1, 2:end, :);
    Bx(1:end-1, 2:end, :) = Bx(1:end-1, 2:end, :) + cf.cozE21.*x1(2:end, 1:end-1, :);
    Bx(2:end, 2:end, :)   = Bx(2:end, 2:end, :)   - cf.cozE21.*x1(1:end-1, 1:end-1, :);
    Bx(1:end-1,1:end-1,:) = Bx(1:end-1, 1:end-1,:)- cf.cozE21.*x1(2:end, 2:end, :);

    Bx(:,2:end,1:end-1)   = Bx(:,2:end,1:end-1)   + cf.coxE23.*x3(:,1:end-1,2:end);
    Bx(:,1:end-1,2:end)   = Bx(:,1:end-1,2:end)   + cf.coxE23.*x3(:,2:end,1:end-1);
    Bx(:,2:end,2:end)     = Bx(:,2:end,2:end)     - cf.coxE23.*x3(:,1:end-1,1:end-1);
    Bx(:,1:end-1,1:end-1) = Bx(:,1:end-1,1:end-1) - cf.coxE23.*x3(:,2:end,2:end);

    %%%%%%%%%%
    Cx = cf.wcen.*x3 + cf.cenE32.*x2 + cf.cenE31.*x1;
    Cx(2:end  , :, :) = Cx(2:end,   :, :) - x3(1:end-1, :, :) .* cf.wcx1 - x1(1:end-1, :, :) .* cf.cx1E31;
    Cx(1:end-1, :, :) = Cx(1:end-1, :, :) - x3(2:end, :,   :) .* cf.wcx2 - x1(2:end, :,   :) .* cf.cx2E31;
    Cx(:, 2:end,   :) = Cx(:, 2:end,   :) - x3(:, 1:end-1, :) .* cf.wcy1 - x2(:, 1:end-1, :) .* cf.cy1E32;
    Cx(:, 1:end-1, :) = Cx(:, 1:end-1, :) - x3(:, 2:end,   :) .* cf.wcy2 - x2(:, 2:end,   :) .* cf.cy2E32;
    Cx(:, :, 2:end  ) = Cx(:, :, 2:end  ) - x3(:, :, 1:end-1) .* cf.wcz1 - x2(:, :, 1:end-1) .* cf.cz1E32 - x1(:, :, 1:end-1) .* cf.cz1E31;
    Cx(:, :, 1:end-1) = Cx(:, :, 1:end-1) - x3(:, :, 2:end  ) .* cf.wcz2 - x2(:, :, 2:end  ) .* cf.cz2E32 - x1(:, :, 2:end  ) .* cf.cz2E31;

    Cx(:,2:end,1:end-1)   = Cx(:,2:end,1:end-1)   + cf.coxE32.*x2(:,1:end-1,2:end);
    Cx(:,1:end-1,2:end)   = Cx(:,1:end-1,2:end)   + cf.coxE32.*x2(:,2:end,1:end-1);
    Cx(:,2:end,2:end)     = Cx(:,2:end,2:end)     - cf.coxE32.*x2(:,1:end-1,1:end-1);
    Cx(:,1:end-1,1:end-1) = Cx(:,1:end-1,1:end-1) - cf.coxE32.*x2(:,2:end,2:end);

    Cx(2:end,  :,1:end-1) = Cx(2:end,  :,1:end-1) + cf.coyE31 .* x1(1:end-1,:,2:end);
    Cx(1:end-1,:,2:end)   = Cx(1:end-1,:,2:end)   + cf.coyE31 .* x1(2:end,  :,1:end-1);
    Cx(2:end  ,:,2:end)   = Cx(2:end  ,:,2:end)   - cf.coyE31 .* x1(1:end-1,:,1:end-1);
    Cx(1:end-1,:,1:end-1) = Cx(1:end-1,:,1:end-1) - cf.coyE31 .* x1(2:end,  :,2:end);
    Bx = -Bx;
end

function [Ax,Bx,Cx] = Ax3Ddiag(cf, x1, x2, x3)
    Ax = cf.ucen.*x1;
    Bx = cf.vcen.*x2;
    Cx = cf.wcen.*x3;
end

function [Ax,Bx,Cx] = Ax3Dsdc(cf, x1, x2, x3)
    Ax = cf.ucen.*x1;
    Ax(2:end  , :, :) = Ax(2:end,   :, :) - x1(1:end-1, :, :) .* cf.ucx1;
    Ax(1:end-1, :, :) = Ax(1:end-1, :, :) - x1(2:end, :,   :) .* cf.ucx2;
    Ax(:, 2:end,   :) = Ax(:, 2:end,   :) - x1(:, 1:end-1, :) .* cf.ucy1;
    Ax(:, 1:end-1, :) = Ax(:, 1:end-1, :) - x1(:, 2:end,   :) .* cf.ucy2;
    Ax(:, :, 2:end  ) = Ax(:, :, 2:end  ) - x1(:, :, 1:end-1) .* cf.ucz1;
    Ax(:, :, 1:end-1) = Ax(:, :, 1:end-1) - x1(:, :, 2:end  ) .* cf.ucz2;
    %%%%%%%%%%
    Bx = cf.vcen.*x2;
    Bx(2:end  , :, :) = Bx(2:end,   :, :) - x2(1:end-1, :, :) .* cf.vcx1;
    Bx(1:end-1, :, :) = Bx(1:end-1, :, :) - x2(2:end, :,   :) .* cf.vcx2;
    Bx(:, 2:end,   :) = Bx(:, 2:end,   :) - x2(:, 1:end-1, :) .* cf.vcy1;
    Bx(:, 1:end-1, :) = Bx(:, 1:end-1, :) - x2(:, 2:end,   :) .* cf.vcy2;
    Bx(:, :, 2:end  ) = Bx(:, :, 2:end  ) - x2(:, :, 1:end-1) .* cf.vcz1;
    Bx(:, :, 1:end-1) = Bx(:, :, 1:end-1) - x2(:, :, 2:end  ) .* cf.vcz2;
    %%%%%%%%%%
    Cx = cf.wcen.*x3;
    Cx(2:end  , :, :) = Cx(2:end,   :, :) - x3(1:end-1, :, :) .* cf.wcx1;
    Cx(1:end-1, :, :) = Cx(1:end-1, :, :) - x3(2:end, :,   :) .* cf.wcx2;
    Cx(:, 2:end,   :) = Cx(:, 2:end,   :) - x3(:, 1:end-1, :) .* cf.wcy1;
    Cx(:, 1:end-1, :) = Cx(:, 1:end-1, :) - x3(:, 2:end,   :) .* cf.wcy2;
    Cx(:, :, 2:end  ) = Cx(:, :, 2:end  ) - x3(:, :, 1:end-1) .* cf.wcz1;
    Cx(:, :, 1:end-1) = Cx(:, :, 1:end-1) - x3(:, :, 2:end  ) .* cf.wcz2;
end


function [ce, xL, xR] = uxx(b1, nxyz, hxyz)
    nx = nxyz(1);
    ny = nxyz(2);
    nz = nxyz(3);
    Nx = nxyz(1) + 1;
    Ny = nxyz(2) + 1;
    Nz = nxyz(3) + 1;
    hx = hxyz(1);
    hy = hxyz(2);
    hz = hxyz(3);
    ja = (hy*hz/hx)/4;

    x3 = zeros(Nx, Ny, Nz);
    x2 = zeros(nx, Ny, Nz);
    x2(:,1:end-1,1:end-1) = x2(:,1:end-1,1:end-1) + b1 ;
    x2(:,2:end,2:end)     = x2(:,2:end,2:end)     + b1;
    x2(:,1:end-1,2:end)   = x2(:,1:end-1,2:end)   + b1;
    x2(:,2:end,1:end-1)   = x2(:,2:end,1:end-1)   + b1;
    x3(2:end,:,:)   = x3(2:end,:,:)   + x2;
    x3(1:end-1,:,:) = x3(1:end-1,:,:) + x2;

    ce = x3 * ja;
    xL = x2 * ja;
    xR = xL;
end

% function [ce, xL, xR] = uyy(b1, nxyz, hxyz)
%     idx = [2,1,3];
%     [ce, xL, xR] = uxx(permute(b1, idx), nxyz(idx), hxyz(idx));
%     ce = permute(ce, idx);
%     xL = permute(xL, idx);
%     xR = permute(xR, idx);
% end

function [ce, xL, xR] = uyy(b1, nxyz, hxyz)
    nx = nxyz(1);
    ny = nxyz(2);
    nz = nxyz(3);
    Nx = nxyz(1) + 1;
    Ny = nxyz(2) + 1;
    Nz = nxyz(3) + 1;
    hx = hxyz(1);
    hy = hxyz(2);
    hz = hxyz(3);
    ja = (hx*hz/hy)/4;

    x3 = zeros(Nx, Ny, Nz);
    x2 = zeros(Nx, ny, Nz);
    x2(1:end-1,:,1:end-1) = x2(1:end-1,:,1:end-1) + b1 ;
    x2(2:end,:,2:end)     = x2(2:end,:,2:end)     + b1;
    x2(1:end-1,:,2:end)   = x2(1:end-1,:,2:end)   + b1;
    x2(2:end,:,1:end-1)   = x2(2:end,:,1:end-1)   + b1;
    x3(:,2:end,:)   = x3(:,2:end,:)   + x2;
    x3(:,1:end-1,:) = x3(:,1:end-1,:) + x2;

    ce = x3 * ja;
    xL = x2 * ja;
    xR = xL;
end

function [ce, xL, xR] = uzz(b1, nxyz, hxyz)
    nx = nxyz(1);
    ny = nxyz(2);
    nz = nxyz(3);
    Nx = nxyz(1) + 1;
    Ny = nxyz(2) + 1;
    Nz = nxyz(3) + 1;
    hx = hxyz(1);
    hy = hxyz(2);
    hz = hxyz(3);
    ja = (hx*hy/hz)/4;

    x3 = zeros(Nx, Ny, Nz);
    x2 = zeros(Nx, Ny, nz);

    x2(1:end-1,1:end-1,:) = x2(1:end-1,1:end-1,:) + b1 ;
    x2(2:end,2:end,:)     = x2(2:end,2:end,:)     + b1;
    x2(1:end-1,2:end,:)   = x2(1:end-1,2:end,:)   + b1;
    x2(2:end,1:end-1,:)   = x2(2:end,1:end-1,:)   + b1;

    x3(:,:,2:end)   = x3(:,:,2:end)   + x2;
    x3(:,:,1:end-1) = x3(:,:,1:end-1) + x2;

    ce = x3 * ja;
    xL = x2 * ja;
    xR = xL;
end

function [ce, xL, xR, yL, yR, co] = uxy(bt, nxyz, hxyz)
    nx = nxyz(1);
    ny = nxyz(2);
    nz = nxyz(3);
    Nx = nxyz(1) + 1;
    Ny = nxyz(2) + 1;
    Nz = nxyz(3) + 1;
    hx = hxyz(1);
    hy = hxyz(2);
    hz = hxyz(3);
    btja = bt*(hz/8.); % ja
    ce = zeros(Nx, Ny, Nz);
    gx = zeros(nx, Ny, Nz);
    gy = zeros(Nx, ny, Nz);
    mbtja = zeros(nx, ny, Nz);
    mbtja(:,:,1:end-1) = mbtja(:,:,1:end-1) + btja;
    mbtja(:,:,2:end)   = mbtja(:,:,2:end)   + btja;
    ce(1:end-1,1:end-1,:) = ce(1:end-1,1:end-1,:) + mbtja;
    ce(2:end,2:end,:)     = ce(2:end,2:end,:)     + mbtja;
    ce(1:end-1,2:end,:)   = ce(1:end-1,2:end,:)   - mbtja;
    ce(2:end,1:end-1,:)   = ce(2:end,1:end-1,:)   - mbtja;
    gx(:,2:end,:)   = gx(:,2:end,:)   + mbtja;
    gx(:,1:end-1,:) = gx(:,1:end-1,:) - mbtja;
    gy(2:end,:,:)   = gy(2:end,:,:)   - mbtja;
    gy(1:end-1,:,:) = gy(1:end-1,:,:) + mbtja;
    % ce,  xL,  xR,  yL, yR, coner
    xL = -gx;
    xR =  gx;
    yL = -gy;
    yR =  gy;
    co =  mbtja;
end

function [ce, xL, xR, yL, yR, co] = uyx(bt, nxyz, hxyz)
    [ce, xL, xR, yL, yR, co] = uxy(bt, nxyz, hxyz);
    xL = -xL;
    xR = -xR;
    yL = -yL;
    yR = -yR;
end

function [ce, xL, xR, yL, yR, co] = uxz(bt, nxyz, hxyz)
    idx = [1,3,2];
    [ce, xL, xR, yL, yR, co] = uxy(permute(bt, idx), nxyz(idx), hxyz(idx));
    ce = permute(ce, idx);
    xL = permute(xL, idx);
    xR = permute(xR, idx);
    yL = permute(yL, idx);
    yR = permute(yR, idx);
    co = permute(co, idx);
end

function [ce, xL, xR, yL, yR, co] = uzx(bt, nxyz, hxyz)
    [ce, xL, xR, yL, yR, co] = uxz(bt, nxyz, hxyz);
    xL = -xL;
    xR = -xR;
    yL = -yL;
    yR = -yR;
end

function [ce, xL, xR, yL, yR, coner] = uyz(bt, nxyz, hxyz)
    nx = nxyz(1);
    ny = nxyz(2);
    nz = nxyz(3);
    Nx = nxyz(1) + 1;
    Ny = nxyz(2) + 1;
    Nz = nxyz(3) + 1;
    hx = hxyz(1);
    hy = hxyz(2);
    hz = hxyz(3);
    btja = bt*(hx/8.); % ja
    ce = zeros(Nx, Ny, Nz);
    gy = zeros(Nx, ny, Nz);
    gz = zeros(Nx, Ny, nz);
    mbtja = zeros(Nx, ny, nz);
    mbtja(1:end-1, :, :) = mbtja(1:end-1, :, :) + btja;
    mbtja(2:end, :, :)   = mbtja(2:end, :, :)   + btja;
    ce(:, 1:end-1, 1:end-1) = ce(:, 1:end-1, 1:end-1) + mbtja;
    ce(:, 2:end, 2:end)     = ce(:, 2:end, 2:end)     + mbtja;
    ce(:, 1:end-1, 2:end)   = ce(:, 1:end-1, 2:end)   - mbtja;
    ce(:, 2:end, 1:end-1)   = ce(:, 2:end, 1:end-1)   - mbtja;
    
    gy(:, :, 2:end) = gy(:, :, 2:end) + mbtja;
    gy(:, :, 1:end-1) = gy(:, :, 1:end-1) - mbtja;
    gz(:, 2:end, : ) = gz(:, 2:end, : ) - mbtja;
    gz(:, 1:end-1,:) = gz(:, 1:end-1,:)   + mbtja;
    coner = mbtja;
    xL = -gy;
    xR = gy;
    yL = -gz;
    yR = gz;
end

function [ce, xL, xR, yL, yR, co] = uzy(bt, nxyz, hxyz)
    [ce, xL, xR, yL, yR, co] = uyz(bt, nxyz, hxyz);
    xL = -xL;
    xR = -xR;
    yL = -yL;
    yR = -yR;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Disclaimer:                                                              %
% The authors reserve all rights but do not guarantee that the code is     %
% free from errors. Furthermore, the authors shall not be liable in any    %
% event caused by the use of the program.                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%