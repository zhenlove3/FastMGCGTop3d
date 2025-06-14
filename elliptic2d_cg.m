function elliptic2d_cg()
    N = 1000;  fprintf('N = %d\n', N);
    maxit = 3;
    tol = 1e-9;
    h = 1 / (N + 1);
    [x, y] = meshgrid(h*(1:N), h*(1:N));
    u_exact = sin(pi * x) .* sin(pi * y) + x.*(x-1).*y.*(y-1);
    f = 2 * pi^2 * sin(pi * x) .* sin(pi * y) - 2*(x.^2 - x + y.^2 - y);
    b = f(:)*(h*h);
    t1 = @() Stencil_based();
    t3 = @() FDM_matrix_free();
    t4 = @() FEM_matrix_free();
    t5 = @() FDM_assemble();
    t6 = @() FEM_assemble();
    fprintf('Average execution time %.6f seconds --- Stencil_based\n',         timeit(t1));
    fprintf('Average execution time %.6f seconds --- FDM_matrix_free\n',                      timeit(t3));
    fprintf('Average execution time %.6f seconds --- FEM_matrix_free\n',                      timeit(t4));
    fprintf('Average execution time %.6f seconds --- FDM_assemble\n',                         timeit(t5));
    fprintf('Average execution time %.6f seconds --- FEM_assemble\n',                         timeit(t6));

    function Stencil_based()
        Afunc = @(x_vec) Ax_stencil(x_vec, N);
        [u_vec, flag, relres, iter] = pcg(Afunc, b, tol, maxit);
        % disp(max(abs(u_exact(:) - u_vec)));  % loo error
    end
    function FDM_matrix_free()
        Ke = Ke2D_reference(1);             %%%%%%%%%%%%%%%% 1
        idx_matrix = generate_quad_indices(N);
        Afunc = @(x_vec) Ax_fem(x_vec, N, idx_matrix, Ke);
        [u_vec, flag, relres, iter] = pcg(Afunc, b, tol, maxit);
        % disp(max(abs(u_exact(:) - u_vec)));  % loo error
    end
    function FEM_matrix_free()
        Ke = Ke2D_reference(1/sqrt(3));     %%%%%%%%%%%%%%%% 1/sqrt(3)
        idx_matrix = generate_quad_indices(N);
        Afunc = @(x_vec) Ax_fem(x_vec, N, idx_matrix, Ke);
        [u_vec, flag, relres, iter] = pcg(Afunc, b, tol, maxit);
        % disp(max(abs(u_exact(:) - u_vec)));  % loo error
    end
    function FDM_assemble()
        idx_matrix = generate_quad_indices(N)'; 
        iK = kron(idx_matrix, ones(1, 4))';
        jK = kron(idx_matrix, ones(4, 1))';
        Ke = Ke2D_reference(1);             %%%%%%%%%%%%%%%% 1
        sK = repmat(Ke(:), (N+1)^2, 1);
        K = sparse(iK(:),jK(:),sK);
        tmp1 =  reshape(1:(N+2)^2, N+2, N+2);
        freedofs = tmp1(2:(N+1), 2:(N+1));
        K = K(freedofs(:), freedofs(:));
        [u_vec, flag, relres, iter] = pcg(K, b, tol, maxit);
        % disp(max(abs(u_exact(:) - u_vec)));  % loo error
    end
    function FEM_assemble()
        idx_matrix = generate_quad_indices(N)'; 
        iK = kron(idx_matrix, ones(1, 4))';
        jK = kron(idx_matrix, ones(4, 1))';
        Ke = Ke2D_reference(1/sqrt(3));     %%%%%%%%%%%%%%%% 1/sqrt(3)
        sK = repmat(Ke(:), (N+1)^2, 1);
        K = sparse(iK(:),jK(:),sK);
        tmp1 =  reshape(1:(N+2)^2, N+2, N+2);
        freedofs = tmp1(2:(N+1), 2:(N+1));
        K = K(freedofs(:), freedofs(:));
        [u_vec, flag, relres, iter] = pcg(K, b, tol, maxit);
        % disp(max(abs(u_exact(:) - u_vec)));  % loo error
    end
end

function Ax_vec = Ax_stencil(x_vec, N, matrix_dot_vector)
    x = reshape(x_vec, N, N);
    Ax = 4 * x;
    Ax(:,2:end) = Ax(:,2:end) - x(:,1:end-1);  
    Ax(:,1:end-1) = Ax(:,1:end-1) - x(:,2:end); 
    Ax(2:end,:) = Ax(2:end,:) - x(1:end-1,:);   
    Ax(1:end-1,:) = Ax(1:end-1,:) - x(2:end,:); 
    Ax_vec = Ax(:);
end

function Ax_vec = Ax_fem(x, N, idx_matrix, Ke)
    xall = zeros(N+2, N+2);
    xall(2:(N+1), 2:(N+1)) = reshape(x, N, N);  % zero boundary conditions
    xall = xall(:);
    Ke_dot_xe = Ke * xall(idx_matrix);  %  ke * xe
    tmp = accumarray(idx_matrix(:), Ke_dot_xe(:), [(N+2)*(N+2), 1]);  % assemble vector
    tmp = reshape(tmp, N+2, N+2);
    tmp2 = tmp(2:(N+1), 2:(N+1));    % zero boundary conditions
    Ax_vec = tmp2(:);
end

function KE = Ke2D_reference(g)
    % g = 1/sqrt(3);  % 2-point Gauss integration
    % g = 1;          % 2-point trapezoidal integration
    w = ones(4, 1);   % weights
    r = [-1, 1, -1, 1];
    s = [-1, -1, 1, 1];
    KE = zeros(4, 4);
    dN = zeros(4, 2); % gradient matrix 
    for q = 1:4
        gx = r(q)*g; gy = s(q)*g;
        dN(:,1) = 0.25 * r .* (1 + s*gy);  % ∂N/∂x
        dN(:,2) = 0.25 * s .* (1 + r*gx);  % ∂N/∂y
        KE = KE + (dN * dN') * w(q);
    end
end

function idx_matrix = generate_quad_indices(N)
    Nx = N + 2; Ny = N + 2; 
    idx = reshape(1:Nx*Ny, Nx, Ny);
    subidx = idx(1:end-1,1:end-1);
    t = subidx(:)';
    idx_matrix = [t; t+1; t+Nx; t+Nx+1];   
end